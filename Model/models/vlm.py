"""VLM-based semantic privacy judgment module (Phase 4).

This module provides:
  1. Multi-view input preparation for VLM analysis
  2. Structured prompt generation
  3. Response parsing
  4. A placeholder `call_vlm_api` method for users to override

Users should subclass VLMPrivacyJudge and implement `call_vlm_api`
to connect to their chosen VLM service (e.g., GPT-4V, Qwen-VL, etc.).
"""

import json
import base64
import io
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import numpy as np
from PIL import Image, ImageDraw


# Valid privacy categories returned by the VLM
PRIVACY_CATEGORIES = [
    'face', 'text', 'document', 'license_plate', 'body',
    'location_marker', 'biometric', 'personal_item', 'none',
]

# Valid risk levels
RISK_LEVELS = ['low', 'medium', 'high', 'critical']

# Valid protection action recommendations
PROTECTION_ACTIONS = ['noise', 'blur', 'mosaic', 'suppress', 'none']

# Default prompt template
DEFAULT_PROMPT = """你是一个图像隐私敏感性分析器。给定一张图像中某个特定区域的多视角信息，请判断该区域是否包含隐私敏感内容。

你会收到以下图像：
1. 原始完整图像
2. 用红色框标记了目标区域位置的完整图像
3. 目标区域的裁剪放大图
4. 目标区域周围的上下文区域

请分析目标区域，并以以下JSON格式回复（不要包含其他内容）：
{
    "category": "<隐私类别: face, text, document, license_plate, body, location_marker, biometric, personal_item, none>",
    "risk_level": "<风险等级: low, medium, high, critical>",
    "privacy_score": <0到1之间的浮点数，表示隐私敏感程度>,
    "recommended_action": "<推荐保护动作: noise, blur, mosaic, suppress, none>",
    "reasoning": "<简要说明判断依据>"
}
"""


def tensor_to_pil(tensor):
    """Convert (C, H, W) or (1, C, H, W) tensor in [0,1] to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_base64(img, fmt='PNG'):
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


class VLMPrivacyJudge:
    """VLM-based semantic privacy judgment.

    Prepares multi-view image inputs and structured prompts for each
    candidate patch, then delegates the actual VLM API call to
    `call_vlm_api` which users must implement.

    Usage:
        class MyVLMJudge(VLMPrivacyJudge):
            def call_vlm_api(self, images_b64, prompt):
                # Call your VLM API here
                response = my_api_call(images_b64, prompt)
                return response  # raw text from VLM

        judge = MyVLMJudge(patch_size=16)
        results = judge.judge_patches(image_tensor, candidates)
    """

    def __init__(self, patch_size=16, context_margin=1, prompt=None):
        """
        Args:
            patch_size:      pixel size of each patch
            context_margin:  number of extra patches around the target for context view
            prompt:          custom prompt template (uses DEFAULT_PROMPT if None)
        """
        self.patch_size = patch_size
        self.context_margin = context_margin
        self.prompt = prompt or DEFAULT_PROMPT

    def prepare_inputs(self, image, candidate):
        """Prepare multi-view images for VLM analysis of one candidate patch.

        Args:
            image:     (1, 3, H, W) tensor in [0, 1]
            candidate: dict with 'row', 'col', 'y0', 'y1', 'x0', 'x1'

        Returns:
            images_b64: list of 4 base64-encoded PNG images
                [original, marked, cropped_patch, context]
        """
        _, _, H, W = image.shape
        ps = self.patch_size
        y0, y1 = candidate['y0'], candidate['y1']
        x0, x1 = candidate['x0'], candidate['x1']

        # 1. Original full image
        pil_orig = tensor_to_pil(image)

        # 2. Full image with red box marking the target patch
        pil_marked = pil_orig.copy()
        draw = ImageDraw.Draw(pil_marked)
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline='red', width=2)

        # 3. Cropped patch (resized for visibility)
        pil_crop = pil_orig.crop((x0, y0, x1, y1))
        crop_size = max(64, ps * 4)  # enlarge small patches
        pil_crop = pil_crop.resize((crop_size, crop_size), Image.NEAREST)

        # 4. Context region (patch + surrounding margin)
        margin = self.context_margin * ps
        ctx_y0 = max(0, y0 - margin)
        ctx_y1 = min(H, y1 + margin)
        ctx_x0 = max(0, x0 - margin)
        ctx_x1 = min(W, x1 + margin)
        pil_ctx = pil_orig.crop((ctx_x0, ctx_y0, ctx_x1, ctx_y1))
        ctx_size = max(64, (ctx_x1 - ctx_x0) * 2)
        pil_ctx = pil_ctx.resize((ctx_size, ctx_size), Image.BILINEAR)

        images_b64 = [
            pil_to_base64(pil_orig),
            pil_to_base64(pil_marked),
            pil_to_base64(pil_crop),
            pil_to_base64(pil_ctx),
        ]
        return images_b64

    def call_vlm_api(self, images_b64, prompt):
        """Call VLM API with prepared inputs. **Override this method.**

        Args:
            images_b64: list of 4 base64-encoded PNG image strings
                        [original, marked, cropped, context]
            prompt:     the analysis prompt text

        Returns:
            response_text: raw text response from VLM (should be JSON)

        Raises:
            NotImplementedError: if not overridden by subclass
        """
        raise NotImplementedError(
            "请继承 VLMPrivacyJudge 并实现 call_vlm_api 方法。\n"
            "Please subclass VLMPrivacyJudge and implement call_vlm_api.\n"
            "示例:\n"
            "  class MyJudge(VLMPrivacyJudge):\n"
            "      def call_vlm_api(self, images_b64, prompt):\n"
            "          return my_api(images_b64, prompt)\n"
        )

    def parse_response(self, response_text):
        """Parse VLM response text into structured result.

        Args:
            response_text: raw JSON string from VLM

        Returns:
            dict with keys: q_score, category, risk_level, action, reasoning
        """
        # Try to extract JSON from response (handle markdown code blocks)
        text = response_text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            # Remove first and last ``` lines
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith('```') and not in_block:
                    in_block = True
                    continue
                elif line.strip() == '```' and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = '\n'.join(json_lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback: return conservative defaults
            return {
                'q_score': 0.5,
                'category': 'none',
                'risk_level': 'medium',
                'action': 'noise',
                'reasoning': f'VLM response parse failed: {response_text[:100]}',
            }

        # Validate and extract fields
        category = data.get('category', 'none')
        if category not in PRIVACY_CATEGORIES:
            category = 'none'

        risk_level = data.get('risk_level', 'medium')
        if risk_level not in RISK_LEVELS:
            risk_level = 'medium'

        q_score = float(data.get('privacy_score', 0.5))
        q_score = max(0.0, min(1.0, q_score))

        action = data.get('recommended_action', 'noise')
        if action not in PROTECTION_ACTIONS:
            action = 'noise'

        reasoning = data.get('reasoning', '')

        return {
            'q_score': q_score,
            'category': category,
            'risk_level': risk_level,
            'action': action,
            'reasoning': reasoning,
        }

    def judge_patches(self, image, candidates):
        """Judge all candidate patches for semantic privacy.

        Args:
            image:      (1, 3, H, W) tensor in [0, 1]
            candidates: list of candidate dicts from OcclusionAnalyzer

        Returns:
            enriched_candidates: same list with added VLM fields:
                'q_score', 'category', 'risk_level', 'action', 'reasoning'
        """
        for cand in candidates:
            images_b64 = self.prepare_inputs(image, cand)
            response_text = self.call_vlm_api(images_b64, self.prompt)
            result = self.parse_response(response_text)
            cand.update(result)

        return candidates


class MockVLMJudge(VLMPrivacyJudge):
    """Mock VLM judge for testing — assigns uniform scores.

    Useful for pipeline testing without a real VLM API.
    All patches get the same default privacy score and action.
    """

    def __init__(self, default_score=0.5, default_action='noise', **kwargs):
        super().__init__(**kwargs)
        self.default_score = default_score
        self.default_action = default_action

    def call_vlm_api(self, images_b64, prompt):
        return json.dumps({
            'category': 'none',
            'risk_level': 'medium',
            'privacy_score': self.default_score,
            'recommended_action': self.default_action,
            'reasoning': 'Mock VLM — testing mode',
        })
