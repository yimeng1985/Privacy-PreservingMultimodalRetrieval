"""VLM-based semantic privacy judgment module (Phase 4).

This module provides:
  1. Multi-view input preparation for VLM analysis
  2. Structured prompt generation
  3. Response parsing
  4. QwenVLMJudge — production implementation via Alibaba DashScope API
  5. MockVLMJudge — placeholder for testing without API calls

Usage (real VLM):
    export DASHSCOPE_API_KEY=sk-xxxx
    # then in config:  vlm.backend: "qwen"

Usage (mock, default):
    # config:  vlm.backend: "mock"
"""

import json
import base64
import io
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


# Valid privacy categories returned by the VLM
PRIVACY_CATEGORIES = [
    'face', 'text', 'document', 'license_plate', 'body',
    'location_marker', 'biometric', 'personal_item', 'none',
]

# Valid risk levels
RISK_LEVELS = ['low', 'medium', 'high', 'critical']

# Valid protection action recommendations
PROTECTION_ACTIONS = ['noise', 'blur', 'mosaic', 'suppress', 'none']

# Default prompt template (per-patch, legacy)
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

# Batch prompt template (one call per image, all candidates at once)
BATCH_PROMPT = """你是一个图像隐私敏感性分析器。下面这张图像中，我用彩色编号方框标注了若干候选区域（编号从1开始）。
请逐个分析每个编号区域是否包含隐私敏感内容（如人脸、文字、证件、车牌、身体部位、地标、生物特征、个人物品等）。

请严格以以下JSON数组格式回复（不要包含其他内容）：
[
  {
    "id": <区域编号，从1开始>,
    "category": "<隐私类别: face, text, document, license_plate, body, location_marker, biometric, personal_item, none>",
    "risk_level": "<风险等级: low, medium, high, critical>",
    "privacy_score": <0到1之间的浮点数>,
    "recommended_action": "<推荐保护动作: noise, blur, mosaic, suppress, none>",
    "reasoning": "<简要判断依据>"
  },
  ...
]
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


class QwenVLMJudge(VLMPrivacyJudge):
    """Production VLM judge using Alibaba DashScope (Qwen-VL) API.

    Uses the OpenAI-compatible endpoint provided by DashScope.
    Requires the ``openai`` Python package (``pip install openai``).

    The API key is read from the ``DASHSCOPE_API_KEY`` environment variable,
    or can be passed directly via the ``api_key`` constructor argument.

    Example:
        judge = QwenVLMJudge(api_key="sk-xxx", model="qwen3.5-plus")
        enriched = judge.judge_patches(image_tensor, candidates)
    """

    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen3.5-plus",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        # Resolve API key: explicit arg > env var
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DashScope API key not found. "
                "Set DASHSCOPE_API_KEY environment variable or pass api_key=."
            )

        # Lazy-init OpenAI client (imported at call time to keep module lightweight)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.DASHSCOPE_BASE_URL,
            )
        return self._client

    def call_vlm_api(self, images_b64, prompt):
        """Call Qwen-VL via DashScope OpenAI-compatible API.

        Sends 4 images (original, marked, cropped, context) together with
        the analysis prompt as a single multi-image user message.

        Args:
            images_b64: list of 4 base64-encoded PNG strings
            prompt:     the analysis prompt text

        Returns:
            response_text: raw text response from VLM
        """
        client = self._get_client()

        # Build multi-image content array
        image_labels = ["原始完整图像", "红框标记目标区域", "目标区域裁剪放大", "上下文区域"]
        content = []
        for label, b64 in zip(image_labels, images_b64):
            content.append({"type": "text", "text": f"【{label}】"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                },
            })
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,  # near-deterministic for consistent scoring
                    max_tokens=512,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                logger.warning(
                    "QwenVLM API call attempt %d/%d failed: %s",
                    attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        # All retries exhausted — return a safe fallback JSON
        logger.error("QwenVLM API failed after %d attempts: %s",
                      self.max_retries, last_error)
        return json.dumps({
            'category': 'none',
            'risk_level': 'medium',
            'privacy_score': 0.5,
            'recommended_action': 'noise',
            'reasoning': f'API call failed: {last_error}',
        })

    # ---- Batch mode: one VLM call per image ----

    def _prepare_batch_image(self, image, candidates):
        """Draw numbered colored boxes on the image for all candidates.

        Returns:
            pil_annotated: PIL Image with all candidates marked
        """
        pil = tensor_to_pil(image)
        draw = ImageDraw.Draw(pil)

        # Use distinct colors so boxes are easy to tell apart
        colors = [
            '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
            '#00FFFF', '#FF8000', '#8000FF', '#00FF80', '#FF0080',
            '#80FF00', '#0080FF', '#FF4040', '#40FF40', '#4040FF',
        ]

        for i, cand in enumerate(candidates):
            color = colors[i % len(colors)]
            y0, y1 = cand['y0'], cand['y1']
            x0, x1 = cand['x0'], cand['x1']
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=2)
            # Draw the index number near top-left of the box (1-based for VLM clarity)
            label = str(i + 1)
            draw.text((x0 + 2, y0 + 1), label, fill=color)

        return pil

    def _parse_batch_response(self, response_text, num_candidates):
        """Parse the VLM batch response into a list of per-patch results.

        Returns:
            list of dicts, one per candidate, indexed by 'id' field.
            Missing entries get conservative defaults.
        """
        text = response_text.strip()
        # Handle markdown code blocks
        if '```' in text:
            lines = text.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('```') and not in_block:
                    in_block = True
                    continue
                elif stripped == '```' and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = '\n'.join(json_lines)

        default_result = {
            'q_score': 0.5, 'category': 'none', 'risk_level': 'medium',
            'action': 'noise', 'reasoning': 'batch parse fallback',
        }

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Batch VLM response parse failed: %s", text[:200])
            return [dict(default_result) for _ in range(num_candidates)]

        if not isinstance(data, list):
            data = [data]

        # Index by 'id' (VLM uses 1-based numbering, convert to 0-based)
        result_map = {}
        for item in data:
            idx = item.get('id')
            if idx is None:
                continue
            idx = int(idx) - 1  # convert 1-based to 0-based
            q_score = float(item.get('privacy_score', 0.5))
            q_score = max(0.0, min(1.0, q_score))

            category = item.get('category', 'none')
            if category not in PRIVACY_CATEGORIES:
                category = 'none'
            risk_level = item.get('risk_level', 'medium')
            if risk_level not in RISK_LEVELS:
                risk_level = 'medium'
            action = item.get('recommended_action', 'noise')
            if action not in PROTECTION_ACTIONS:
                action = 'noise'

            result_map[idx] = {
                'q_score': q_score,
                'category': category,
                'risk_level': risk_level,
                'action': action,
                'reasoning': item.get('reasoning', ''),
            }

        # Fill missing entries with defaults
        results = []
        for i in range(num_candidates):
            results.append(result_map.get(i, dict(default_result)))
        return results

    def judge_patches(self, image, candidates):
        """Batch judge: annotate ALL candidates on one image, call VLM once.

        This overrides the base per-patch loop and reduces API calls from
        N to 1 per image.
        """
        if not candidates:
            return candidates

        # Prepare annotated image with all candidates
        pil_annotated = self._prepare_batch_image(image, candidates)
        b64_annotated = pil_to_base64(pil_annotated)
        b64_orig = pil_to_base64(tensor_to_pil(image))

        # Build content: original + annotated with numbered boxes
        client = self._get_client()
        content = [
            {"type": "text", "text": "【原始图像】"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_orig}"}},
            {"type": "text", "text": f"【标注图像】（共 {len(candidates)} 个编号区域）"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_annotated}"}},
            {"type": "text", "text": BATCH_PROMPT},
        ]
        messages = [{"role": "user", "content": content}]

        # Call API with retry
        last_error = None
        response_text = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                )
                response_text = response.choices[0].message.content
                break
            except Exception as e:
                last_error = e
                logger.warning("Batch VLM call attempt %d/%d failed: %s",
                               attempt, self.max_retries, e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        if response_text is None:
            logger.error("Batch VLM failed after %d attempts: %s",
                          self.max_retries, last_error)
            # Fallback: assign default scores
            for cand in candidates:
                cand.update({
                    'q_score': 0.5, 'category': 'none', 'risk_level': 'medium',
                    'action': 'noise', 'reasoning': f'API failed: {last_error}',
                })
            return candidates

        # Parse batch response
        results = self._parse_batch_response(response_text, len(candidates))
        for cand, result in zip(candidates, results):
            cand.update(result)

        return candidates
