import os
from typing import Optional, List, Dict, Any

import torch
from PIL import Image

from .base_adapter import BaseAdapter
from utils.constants import WEBQA_TASK, HEADING_OCR_TASK, ACTION_GROUND_TASK

import cv2
import numpy as np


# =================== Shared helpers =================== #

def crop_normalized(img: Image.Image, bbox):
    """
    bbox: [x0, y0, x1, y1] in normalized coordinates [0, 1].
    Returns a cropped image, resized back to the original size.

    NOTE: With a regular k×k grid plus symmetric margin, the crop
    keeps (approximately) the same aspect ratio as the original,
    so resizing back to (W, H) is fine.
    """
    w, h = img.size
    x0, y0, x1, y1 = bbox

    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))

    left = int(x0 * w)
    top = int(y0 * h)
    right = int(x1 * w)
    bottom = int(y1 * h)

    if right <= left or bottom <= top:
        # Degenerate box → return original image
        return img

    cropped = img.crop((left, top, right, bottom))
    # Resize back to original resolution to keep text legible
    return cropped.resize((w, h), Image.BICUBIC)


def build_zone_mapping(grid_size: int):
    """
    Build:
      - zone_to_bbox: mapping from zone name → [x0,y0,x1,y1]
      - synonyms: optional extra names that alias canonical zones (for k=3).

    Canonical names are always 'R<i>C<j>'.

    For k = 3, we also accept TL/TM/... as synonyms mapping to R1C1...R3C3.
    """
    k = grid_size
    zone_to_bbox: Dict[str, List[float]] = {}
    synonyms: Dict[str, str] = {}

    cell_w = 1.0 / k
    cell_h = 1.0 / k

    for row in range(1, k + 1):
        for col in range(1, k + 1):
            x0 = (col - 1) * cell_w
            x1 = col * cell_w
            y0 = (row - 1) * cell_h
            y1 = row * cell_h
            name = f"R{row}C{col}"
            zone_to_bbox[name] = [x0, y0, x1, y1]

    # Add TL/TM/... synonyms only when k = 3
    if k == 3:
        synonyms = {
            "TL": "R1C1",
            "TM": "R1C2",
            "TR": "R1C3",
            "ML": "R2C1",
            "MM": "R2C2",
            "MR": "R2C3",
            "BL": "R3C1",
            "BM": "R3C2",
            "BR": "R3C3",
        }

    return zone_to_bbox, synonyms


# ---------- Heading OCR helpers ---------- #

def clean_heading(raw: str) -> str:
    """
    Heuristic cleaner for heading OCR.

    - Strips quotes and whitespace.
    - If there's a colon, prefers the part after the colon
      (e.g., 'The main heading is: Hello World' -> 'Hello World').
    """
    if raw is None:
        return ""

    text = raw.strip().strip('"').strip("'")

    # If model output is like "The main heading is: XYZ"
    if ":" in text:
        before, after = text.split(":", 1)
        after = after.strip()
        if after:  # only use if non-empty
            text = after

    return text.strip()


def _extract_top_heading_from_summary(summary_text: str) -> str:
    """
    Given the heading-summary output (list of headings),
    extract the first heading candidate.

    Expected format (but we stay lenient):
      1. Some Heading Text
      2. Another Heading

    We take the first non-empty line, strip leading '1.' etc.
    """
    if not summary_text:
        return ""

    lines = [ln.strip() for ln in summary_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    first = lines[0]

    # Remove leading numbering like '1.' or '1)'
    # e.g. "1. Hello World" -> "Hello World"
    if first[0].isdigit():
        # find first space or dot after digits
        i = 0
        while i < len(first) and first[i].isdigit():
            i += 1
        # skip dot or ) and following spaces
        while i < len(first) and first[i] in [".", ")", " "]:
            i += 1
        first = first[i:].strip()

    return clean_heading(first)


# =================== VisualCoTAgent (original WebQA design) =================== #

class VisualCoTAgent:
    """
    Fixed-depth multi-view WebQA agent with k×k grid and margin cropping:

      - Views: full page + up to `max_crops` successive crops.
      - At each view:
          1) ANSWER-ONLY call that also outputs info_visible: YES/NO.
          2) If not last view: CROP policy (must pick exactly one grid cell).

      - Cropping:
          - Policy selects a cell (R<i>C<j>).
          - We expand that cell's bbox by a margin fraction of cell width/height,
            then crop + resize back to original image size.
          - This helps when the relevant text sits near the boundary between cells.

      - Final answer rule (last-YES-wins with graceful fallback):
          - If there exists at least one candidate with info_visible == YES,
              pick the **last** such candidate (later crops override earlier guesses).
          - If **no** candidate has info_visible == YES,
              fall back to the **first** candidate (the earliest global context guess).
    """

    def __init__(
        self,
        processor,
        model,
        *,
        max_new_tokens: int = 48,
        save_dir: Optional[str] = None,
        grid_size: int = 3,
        max_crops: int = 1,
        margin_frac_of_cell: float = 0.2,
    ):
        """
        Args:
            processor, model: pre-loaded LLaVA processor/model (we reuse the adapter's).
            max_new_tokens: max tokens per generation.
            save_dir: if not None, directory to save crops for debugging.
            grid_size: k for k×k grid (k >= 2).
            max_crops: maximum number of successive crops (views = 1 + max_crops).
            margin_frac_of_cell: how much to expand each chosen cell on each side,
                as a fraction of the cell size (e.g. 0.2 = expand by 20% of cell width/height).
        """
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")

        self.k = grid_size
        self.zone_to_bbox, self.synonyms = build_zone_mapping(self.k)

        self.max_crops = max_crops
        self.max_new_tokens = max_new_tokens
        self.margin_frac_of_cell = max(0.0, margin_frac_of_cell)

        self.processor = processor
        self.model = model

        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            print(f"[agent] Will save step images to: {self.save_dir}")

    # ------------------- Internal helpers ------------------- #

    def _zones_description(self) -> str:
        k = self.k
        desc = [
            f"The page is divided into a {k}x{k} grid of zones.",
            "Cells are named R<i>C<j>, where i is the row index (top=1, bottom=k),",
            "and j is the column index (left=1, right=k). For example, "
            f"R1C1 is top-left, R1C{k} is top-right, R{k}C1 is bottom-left, "
            f"and R{k}C{k} is bottom-right.",
        ]
        if k == 3:
            desc.append(
                "For k=3, you may also use the synonyms TL, TM, TR, ML, MM, MR, BL, BM, BR, "
                "where TL=R1C1, TM=R1C2, TR=R1C3, ML=R2C1, MM=R2C2, MR=R2C3, "
                "BL=R3C1, BM=R3C2, BR=R3C3."
            )
        return "\n".join(desc) + "\n"

    def _canonical_zone_name(self, zone: str) -> str:
        z = zone.upper()
        if z in self.zone_to_bbox:
            return z
        if z in self.synonyms:
            return self.synonyms[z]
        raise ValueError(f"Unknown zone name '{zone}' for grid_size={self.k}")

    def _expand_bbox_with_margin(self, bbox):
        """
        Expand a cell bbox by margin_frac_of_cell on each side,
        where the margin is expressed as a fraction of a single cell's
        width/height.
        """
        x0, y0, x1, y1 = bbox
        cell_w = 1.0 / self.k
        cell_h = 1.0 / self.k

        dx = self.margin_frac_of_cell * cell_w
        dy = self.margin_frac_of_cell * cell_h

        x0_exp = max(0.0, x0 - dx)
        y0_exp = max(0.0, y0 - dy)
        x1_exp = min(1.0, x1 + dx)
        y1_exp = min(1.0, y1 + dy)

        return [x0_exp, y0_exp, x1_exp, y1_exp]

    def _llava_call(
        self,
        image: Image.Image,
        instruction_text: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Single LLaVA call → return raw decoded text.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction_text},
                    {"type": "image"},
                ],
            },
        ]

        prompt_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            images=image,
            text=prompt_text,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=False,  # greedy
            )

        input_ids = inputs["input_ids"]
        generated_ids = output_ids[0, input_ids.shape[1]:]

        full_text = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        return full_text

    # ------------------- ANSWER-ONLY prompt & parsing ------------------- #

    def _build_answer_instruction(
        self,
        question: str,
        view_tag: str,
        view_index: int,
        total_views: int,
    ) -> str:
        """
        Instruction for ANSWER-ONLY call:
        - General for any WebQA.
        - info_visible must be STRICT: YES only when the key info is clearly present and legible.
        """
        return (
            "You are a visual question answering model looking at a web page screenshot.\n"
            "Your task is to answer the user's question as accurately as possible based ONLY on this current view.\n\n"
            "You MUST respond in EXACTLY one line with the following format:\n"
            " ANSWER: <final answer> | info_visible: YES or NO | reason: <very short reason>\n\n"
            "The flag info_visible is VERY STRICT:\n"
            "- Set info_visible: YES ONLY if the key information needed to answer the question is clearly present,\n"
            "  fully readable, and directly visible in this view (for example, you can literally read the number,\n"
            "  name, or date that answers the question).\n"
            "- If you are guessing, only see partial context, or cannot clearly see the exact required value, you MUST\n"
            "  set info_visible: NO, even if you still attempt to give your best guess in the ANSWER.\n"
            "- When in doubt, choose info_visible: NO.\n\n"
            "Do not talk about cropping or future steps. Do not output JSON.\n\n"
            f"Question: {question}\n"
            f"Current view tag: {view_tag}\n"
            f"View index: {view_index} of {total_views-1}.\n"
        )

    def _split_answer_info_reason(self, line: str):
        """
        Parse line of form:
          ANSWER: <ans> | info_visible: YES/NO | reason: <reason>
        Be tolerant if the pattern is slightly off.
        """
        text = line.strip()

        # Strip leading 'ANSWER:'
        if "ANSWER:" in text:
            _, rest = text.split("ANSWER:", 1)
            text = rest.strip()

        # Defaults
        answer = text
        info_visible = "UNKNOWN"
        reason = ""

        parts = [p.strip() for p in text.split("|")]

        if parts:
            answer = parts[0].strip()

        for p in parts[1:]:
            low = p.lower()
            if "info_visible" in low:
                if ":" in p:
                    _, v = p.split(":", 1)
                    info_visible = v.strip().upper()
            elif "reason" in low:
                if ":" in p:
                    _, v = p.split(":", 1)
                    reason = v.strip()

        return answer, info_visible, reason

    def _parse_answer_candidate(self, raw_text: str) -> Dict[str, Any]:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            return {
                "answer": "UNCERTAIN",
                "info_visible": "UNKNOWN",
                "reason": "Model returned empty output.",
                "raw_full": raw_text,
                "raw_line": "",
            }

        line = lines[-1]
        ans, info_visible, reason = self._split_answer_info_reason(line)
        return {
            "answer": ans,
            "info_visible": info_visible,
            "reason": reason,
            "raw_full": raw_text,
            "raw_line": line,
        }

    # ------------------- CROP policy prompt & parsing ------------------- #

    def _build_policy_instruction(
        self,
        question: str,
        view_tag: str,
        view_index: int,
        total_views: int,
    ) -> str:
        """
        Instruction for CROP policy:
        - It MUST choose exactly one CROP_ZONE.
        - No NO_CROP / STOP option.
        - Does NOT see previous answers.
        """
        zones_desc = self._zones_description()
        return (
            "You are a visual cropping policy for a web page screenshot.\n"
            "You see the current view (full page or a previous crop) and the question.\n"
            "Your ONLY job is to choose ONE grid cell to zoom into next.\n\n"
            + zones_desc +
            "You MUST respond in exactly ONE line with this format:\n"
            " CROP_ZONE R<i>C<j> | reason: <why this cell is the single best region to zoom>\n\n"
            "Guidelines:\n"
            "- Pretend this next crop is your FINAL zoom; choose the one region that is most likely to contain\n"
            "  missing or more detailed information needed to answer the question.\n"
            "- Focus on panels, text blocks, icons, or numbers that directly relate to the question.\n"
            "- Do NOT output NO_CROP or STOP. You must always choose a CROP_ZONE.\n\n"
            f"Question: {question}\n"
            f"Current view tag: {view_tag}\n"
            f"View index: {view_index} of {total_views-1}.\n"
        )

    def _parse_policy_line(self, raw_text: str) -> Dict[str, Any]:
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            return {
                "action": "ERROR",
                "reason": "Empty policy output.",
                "raw_full": raw_text,
                "raw_line": "",
            }

        line = lines[-1]

        if not line.startswith("CROP_ZONE"):
            return {
                "action": "ERROR",
                "reason": f"Policy did not start with CROP_ZONE: {line}",
                "raw_full": raw_text,
                "raw_line": line,
            }

        before, *after = line.split("|", 1)
        parts = before.split()
        if len(parts) < 2:
            return {
                "action": "ERROR",
                "reason": f"Missing zone in policy line: {line}",
                "raw_full": raw_text,
                "raw_line": line,
            }

        zone_raw = parts[1].strip()
        try:
            zone_canonical = self._canonical_zone_name(zone_raw)
            bbox = self.zone_to_bbox[zone_canonical]
        except Exception as e:
            return {
                "action": "ERROR",
                "reason": f"Invalid zone '{zone_raw}': {e}",
                "raw_full": raw_text,
                "raw_line": line,
            }

        reason = ""
        if after and "reason:" in after[0]:
            reason = after[0].split("reason:", 1)[1].strip()

        return {
            "action": "CROP",
            "zone": zone_canonical,
            "zone_raw": zone_raw,
            "bbox": bbox,
            "reason": reason,
            "raw_full": raw_text,
            "raw_line": line,
        }

    # ------------------- Main loop with last-YES-wins aggregation ------------------- #

    def run_chain(
        self,
        image: Image.Image,
        question: str,
        crop_dup_epsilon: float = 1e-3,
    ):
        """
        Main loop:

          - Number of views = 1 + max_crops:
              view 0: full image
              view 1: crop after policy_0
              view 2: crop after policy_1
              ...

          - At each view v:
              1) ANSWER-ONLY → candidate answer_v (with strict info_visible).
              2) If v < last_view: POLICY → CROP_ZONE → next view image
                 (bbox is expanded with margin before cropping).

          - Final answer:
              - If any candidate has info_visible in {YES, TRUE, VISIBLE}:
                    pick the *last* such candidate (later views override).
              - Else:
                    pick the *first* candidate (earliest global context guess).
        """
        original_img = image
        current_img = image

        history: List[Dict[str, Any]] = []
        answer_candidates: List[Dict[str, Any]] = []

        last_crop_bbox = None

        num_views = 1 + max(self.max_crops, 0)

        # Save the initial full image once for visualization
        if self.save_dir is not None:
            init_path = os.path.join(self.save_dir, "step_00_input.png")
            original_img.save(init_path)
            print(f"[agent] Saved initial full image to {init_path}")

        for view_idx in range(num_views):
            view_tag = "full" if view_idx == 0 else f"crop_{view_idx}"

            # ---------- 1) ANSWER-ONLY on current view ----------
            ans_instr = self._build_answer_instruction(
                question=question,
                view_tag=view_tag,
                view_index=view_idx,
                total_views=num_views,
            )
            raw_ans = self._llava_call(current_img, ans_instr)
            print("\n[agent] ---- FULL ASSISTANT OUTPUT (ANSWER-ONLY) ----")
            print(raw_ans)
            print("[agent] --------------------------------------------")

            cand = self._parse_answer_candidate(raw_ans)
            cand["step"] = view_idx
            cand["view"] = view_tag
            answer_candidates.append(cand)
            print(
                f"[agent] Answer-only candidate at view {view_idx} ({view_tag}): "
                f"answer={cand['answer']!r}, info_visible={cand['info_visible']}, "
                f"reason={cand['reason']!r}"
            )

            # If this is the last view, no further crops
            if view_idx == num_views - 1:
                break

            # ---------- 2) CROP policy for NEXT view ----------
            policy_instr = self._build_policy_instruction(
                question=question,
                view_tag=view_tag,
                view_index=view_idx,
                total_views=num_views,
            )
            raw_policy = self._llava_call(current_img, policy_instr)
            print("\n[agent] ---- FULL ASSISTANT OUTPUT (POLICY) ----")
            print(raw_policy)
            print("[agent] ----------------------------------------")

            policy_obj = self._parse_policy_line(raw_policy)
            print(f"[agent] Parsed policy at view {view_idx}: {policy_obj}")

            if policy_obj["action"] != "CROP":
                history.append({
                    "action": "POLICY_ERROR",
                    "view": view_tag,
                    "reason": policy_obj.get("reason", ""),
                    "raw_line": policy_obj.get("raw_line", ""),
                })
                print("[agent] Policy error; stopping further crops.")
                break

            base_bbox = policy_obj["bbox"]
            # Expand with margin
            bbox = self._expand_bbox_with_margin(base_bbox)

            zone = policy_obj.get("zone", "?")
            reason = policy_obj.get("reason", "")

            # Avoid degenerate loop: if new crop is nearly identical to last, stop
            if last_crop_bbox is not None:
                diff = sum(abs(a - b) for a, b in zip(bbox, last_crop_bbox))
                if diff < crop_dup_epsilon:
                    history.append({
                        "action": "CROP_DUP",
                        "from_view": view_tag,
                        "zone": zone,
                        "bbox": bbox,
                        "reason": reason,
                    })
                    print("[agent] New CROP bbox almost identical to previous; "
                          "avoiding infinite loop; stopping further crops.")
                    break

            history.append({
                "action": "CROP",
                "from_view": view_tag,
                "to_view": f"crop_{view_idx+1}",
                "zone": zone,
                "bbox": bbox,
                "reason": reason,
            })
            print(f"[agent] View {view_idx}: CROP_ZONE {zone} bbox={bbox} reason={reason}")

            # Apply crop → next view
            current_img = crop_normalized(current_img, bbox)
            last_crop_bbox = bbox

            if self.save_dir is not None:
                crop_path = os.path.join(self.save_dir, f"step_{view_idx:02d}_crop.png")
                current_img.save(crop_path)
                print(f"[agent] Saved view-{view_idx} cropped image to {crop_path}")

        # ---------- Aggregation: last-YES-wins with first-candidate fallback ----------
        print("\n[agent] ===== ALL ANSWER CANDIDATES =====")
        for cand in answer_candidates:
            print(
                f"  step={cand['step']}, view={cand['view']}, "
                f"info_visible={cand['info_visible']}, answer={cand['answer']!r}, "
                f"reason={cand['reason']!r}"
            )
        print("[agent] ==================================")

        if not answer_candidates:
            print("[agent] No answer candidates collected; returning None.")
            return {
                "history": history,
                "final_answer": None,
                "answer_candidates": [],
            }

        # Find indices where info_visible is clearly YES-like
        yes_indices = [
            i for i, c in enumerate(answer_candidates)
            if c.get("info_visible", "UNKNOWN") in ("YES", "TRUE", "VISIBLE")
        ]

        if yes_indices:
            best_idx = yes_indices[-1]   # last YES wins
        else:
            best_idx = 0  # fallback: first candidate when everything is NO/UNKNOWN

        best_cand = answer_candidates[best_idx]
        final_answer = best_cand["answer"]

        print(
            f"[agent] Selected final candidate: step={best_cand['step']}, "
            f"view={best_cand['view']}, info_visible={best_cand['info_visible']}, "
            f"answer={final_answer!r}"
        )

        return {
            "history": history,
            "final_answer": final_answer,
            "answer_candidates": answer_candidates,
        }


# =================== LlavaHFAdapter (WebQA + heading OCR) =================== #

class LlavaHFAdapter(BaseAdapter):
    """
    HuggingFace LLaVA(-Next) adapter with optional VisualCoTAgent for WebQA.

    - For WEBQA_TASK and use_agent=True:
        Uses VisualCoTAgent (multi-view cropping) with the raw question,
        then refines the chosen final answer to a minimal phrase.
    - For HEADING_OCR_TASK (with use_agent=True):
        Uses full-page heading summary + top-strip crop + a model-based
        decision over the final heading.
    - For all other tasks:
        Simple one-shot LLaVA call with (prompt, image).
    """

    def __init__(
        self,
        model,
        processor,
        use_agent: bool = False,
        agent: Optional[VisualCoTAgent] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        agent_grid_size: int = 3,
        agent_max_crops: int = 1,
        agent_margin_frac_of_cell: float = 0.2,
        agent_save_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model=model, tokenizer=None)
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Disable cache if present
        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.model.eval()

        # Agent setup
        self.use_agent = use_agent
        if use_agent:
            if agent is not None:
                self.agent = agent
            else:
                self.agent = VisualCoTAgent(
                    processor=self.processor,
                    model=self.model,
                    max_new_tokens=max_new_tokens,
                    grid_size=agent_grid_size,
                    max_crops=agent_max_crops,
                    margin_frac_of_cell=agent_margin_frac_of_cell,
                    save_dir=agent_save_dir,
                )
        else:
            self.agent = None

    # ---- core one-shot LLaVA call ----
    def _llava_generate(self, query: str, image: Image.Image) -> str:
        """Single multimodal generation using HF LLaVA (no cropping)."""
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        gen_kwargs = dict(max_new_tokens=self.max_new_tokens)
        if self.temperature is None or self.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]

        text = self.processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        return text
    
    # ================== Action grounding helpers (letter-labeled boxes) ================== #

    def _scan_region_for_labels(
        self,
        region_image: Image.Image,
        region_tag: str,
        instruction: str,
    ) -> str:
        """
        Run on a cropped region (e.g. TL / TR / BL / BR).

        Asks the model to:
          - find red, letter-labeled boxes,
          - read / summarize the content INSIDE each red box.
        """
        prompt = (
            "You are looking at ONE PART of a web page screenshot.\n"
            "In this dataset, CANDIDATE ACTION REGIONS are drawn as rectangles with a BRIGHT RED BORDER.\n"
            "Each of these red rectangles has a SINGLE WHITE CAPITAL LETTER label (A, B, C, ...)\n"
            "placed near one of its corners, usually with a dark or black highlight.\n\n"
            "Your tasks in THIS REGION ONLY:\n"
            "1. Find all red, letter-labeled rectangles that are at least partly visible.\n"
            "2. For each letter label, describe what is INSIDE that red box and what its main purpose is\n"
            "  \n\n"
            "Important rules:\n"
            "- Only treat a letter as a label if it is a SINGLE capital letter (A–Z) directly associated\n"
            "  with a red rectangular border.\n"
            "- Do NOT invent labels that are not clearly visible in this region.\n"
            "- Ignore letters that are part of normal text, words, paragraphs, or logos.\n"
            "- When you describe a label, focus on the UI element INSIDE the red box,\n"
            "  not the surrounding page.\n"
            "- If a red box is partly cut off by the crop, still include it if you can see enough\n"
            "  to guess its role.\n"
            "- Never write descriptions like 'NONE', 'no label', or 'no content' for a label.\n"
            "  If you truly see no red, letter-labeled rectangles at all, use the special output\n"
            "  described below.\n\n"
            "You will later be asked to choose ONE label that best satisfies this instruction:\n"
            f"INSTRUCTION: {instruction}\n\n"
            "Output format (MUST follow one of these exactly):\n"
            "1) If you see one or more valid red, letter-labeled rectangles in this region,\n"
            "   output ONE line per label, in any order:\n"
            "      LETTER: short description (<= 10 words)\n"
            "   where LETTER is a single capital letter A–Z with NO brackets or extra symbols.\n"
            "   For example:\n"
            "      A: a button that ....\n"
            "   Each label must appear at most once.\n\n"
            "2) If you do NOT see any red, letter-labeled rectangles in this region,\n"
            "   output exactly one line with:\n"
            "      NONE\n\n"
            "Do NOT add any other text, explanations, or JSON.\n"
            f"Current region tag: {region_tag}\n"
        )



        return self._llava_generate(prompt, region_image)


    def _parse_region_labels(
        self,
        raw_text: str,
        region_tag: str,
    ) -> List[Dict[str, str]]:
        """
        Parse output of _scan_region_for_labels into a list of dicts:
          { 'label': 'A', 'description': '...', 'region': 'TL' }
        """
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        if not lines:
            return []

        # If the model explicitly says NONE anywhere → treat as no labels
        joined = " ".join(lines).lower()
        if "none" in joined and all(len(ln) <= 8 for ln in lines):
            return []

        results: List[Dict[str, str]] = []
        for ln in lines:
            # Expect something like "A: description"
            if ":" not in ln:
                continue
            left, right = ln.split(":", 1)
            left = left.strip()
            right = right.strip()
            if not left or len(left) != 1 or not left.isalpha():
                continue
            label = left.upper()
            desc = right
            if not desc:
                continue
            results.append(
                {
                    "label": label,
                    "description": desc,
                    "region": region_tag,
                }
            )
        return results

    def _collect_action_candidates(
        self,
        image: Image.Image,
        instruction: str,
    ) -> Dict[str, str]:
        """
        Divide the page into a 2x2 grid (TL, TR, BL, BR), scan each region for labels,
        and aggregate descriptions per label.

        Returns:
            candidates: dict[label] = merged_description
        """
        # 2x2 quadrants in normalized coords
        regions = {
            "TL": [0.0, 0.0, 0.5, 0.5],
            "TR": [0.5, 0.0, 1.0, 0.5],
            "BL": [0.0, 0.5, 0.5, 1.0],
            "BR": [0.5, 0.5, 1.0, 1.0],
        }

        candidates: Dict[str, str] = {}

        for tag, bbox in regions.items():
            region_img = crop_normalized(image, bbox)
            print(f"[ActionGround][scan] Region={tag}, bbox={bbox}")
            raw = self._scan_region_for_labels(region_img, tag, instruction)
            print(f"[ActionGround][scan] raw output ({tag}):")
            print(raw)

            parsed = self._parse_region_labels(raw, tag)
            print(f"[ActionGround][scan] parsed labels ({tag}): {parsed}")

            for item in parsed:
                lbl = item["label"]
                desc = item["description"]
                # If the label appears multiple times (e.g. overlapping crops),
                # keep the longer / more informative description.
                if lbl not in candidates or len(desc) > len(candidates[lbl]):
                    candidates[lbl] = desc

        print("[ActionGround] aggregated candidates:")
        for lbl, desc in candidates.items():
            print(f"  {lbl}: {desc!r}")
        return candidates

    def _select_action_label(
        self,
        image: Image.Image,
        instruction: str,
        candidates: Dict[str, str],
    ) -> str:
        """
        Given a dict[label] = description, ask the model to pick exactly one label.
        The call sees the FULL PAGE screenshot again, plus the candidate summaries
        and the natural-language instruction.
        """
        if not candidates:
            print("[ActionGround] No candidates found; falling back to 'A'.")
            return "A"  # trivial fallback

        # Build candidate list string
        lines = [f"{lbl}: {desc}" for lbl, desc in sorted(candidates.items())]
        labels_str = "\n".join(lines)
        label_set_str = ", ".join(sorted(candidates.keys()))

        prompt = (
            "You are looking at the FULL screenshot of a web page.\n"
            "Several UI regions on this page are highlighted by rectangles with a BRIGHT RED BORDER.\n"
            "Each such red box has a SINGLE CAPITAL LETTER label (A, B, C, ...).\n\n"
            "From previous steps, we extracted a short description of what is inside each\n"
            "red, letter-labeled box:\n"
            f"{labels_str}\n\n"
            "User instruction:\n"
            f"{instruction}\n\n"
            "Your job:\n"
            "- Using BOTH the screenshot and the descriptions above, decide which ONE\n"
            "  labeled red box the user should select in order to follow the instruction.\n"
            "- Think about what the user wants to do (e.g., open an article, use a search box,\n"
            "  open a menu, view an animal card, etc.) and match it to the most appropriate region.\n"
            "- You MUST choose exactly ONE label from the provided set.\n\n"
            "Output format (MUST follow exactly):\n"
            "  ANSWER: <LABEL> | reason: <very short reason>\n\n"
            f"where <LABEL> MUST be one of: {label_set_str}.\n"
            "Do NOT output any other text, JSON, or multiple labels.\n"
        )

        raw = self._llava_generate(prompt, image)
        print("[ActionGround][select] raw output:")
        print(raw)

        # Parse the last non-empty line
        lines_out = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if not lines_out:
            chosen = sorted(candidates.keys())[0]
            print(f"[ActionGround][select] Empty output; fallback to {chosen}")
            return chosen

        last = lines_out[-1]
        if "ANSWER:" in last:
            _, rest = last.split("ANSWER:", 1)
            last = rest.strip()

        # Strip everything after first '|' if present
        if "|" in last:
            last = last.split("|", 1)[0].strip()

        # We expect the remaining piece to contain a single label
        label_char = ""
        for ch in last:
            if ch.isalpha():
                label_char = ch.upper()
                break

        if not label_char:
            chosen = sorted(candidates.keys())[0]
            print(f"[ActionGround][select] No valid label parsed; fallback to {chosen}")
            return chosen

        if label_char not in candidates:
            chosen = sorted(candidates.keys())[0]
            print(
                f"[ActionGround][select] Parsed label {label_char!r} not in candidates; "
                f"fallback to {chosen}"
            )
            return chosen

        print(f"[ActionGround][select] Final chosen label: {label_char}")
        return label_char


    def _run_action_grounding(
        self,
        image: Image.Image,
        instruction: str,
    ) -> str:
        """
        Full action grounding pipeline:

          1) Split page into 2x2 regions (TL, TR, BL, BR).
          2) For each region, detect letter labels + descriptions.
          3) Aggregate candidates by label.
          4) Ask the model to pick exactly ONE label.

        Returns:
            A single uppercase letter (A–Z) as the predicted action region label.
        """
        print("\n[ActionGround] ====== START ACTION GROUNDING ======")
        candidates = self._collect_action_candidates(image, instruction)

        if not candidates:
            # As a very last resort, use a one-shot guess based on the full page.
            # But still clamp to 'A' for safety (dataset may expect a letter).
            print("[ActionGround] No candidates after scanning; return 'A' as trivial guess.")
            return "A"

        chosen_label = self._select_action_label(image, instruction, candidates)
        print(f"[ActionGround] ====== FINAL ACTION LABEL: {chosen_label} ======")
        return chosen_label


    # ---- WebQA answer refiner ----
    def _refine_answer(
        self,
        question: str,
        raw_answer: str,
        image: Image.Image,
    ) -> str:
        """
        Use LLaVA once more to trim the chosen answer
        to the shortest phrase that directly answers the question.
        """
        if not raw_answer or not raw_answer.strip():
            return ""

        prompt = (
            "You are given a QUESTION and a CANDIDATE ANSWER that were extracted from a web page.\n"
            "Your job is to trim the candidate answer to the SHORTEST phrase that still correctly\n"
            "answers the question.\n\n"
            f"QUESTION: {question}\n"
            f"CANDIDATE ANSWER: {raw_answer}\n\n"
            "Return ONLY the trimmed answer phrase.\n"
            "Do NOT add any explanation or extra words.\n"
        )

        text = self._llava_generate(prompt, image)

        # Light cleaning: remove surrounding quotes/whitespace.
        trimmed = text.strip().strip('"').strip("'").strip()
        if not trimmed:
            return raw_answer
        return trimmed

    # ---- Heading OCR summary call ----
    def _heading_summary_call(self, image: Image.Image) -> str:
        """
        Ask the model to list main headings on the page (full screenshot).
        We want exact literal texts, not paraphrases.
        """
        prompt = (
            "You are looking at a webpage screenshot.\n"
            "List up to 3 main heading texts you see, from the most prominent to less prominent.\n\n"
            "Important:\n"
            "- Copy the heading texts exactly as they appear on the page.\n"
            "- Do NOT paraphrase.\n"
            "- Do NOT add any explanation.\n\n"
            "Format:\n"
            "1. <heading 1>\n"
            "2. <heading 2>\n"
            "3. <heading 3>\n"
        )

        return self._llava_generate(prompt, image)
    

    # ---- Heading OCR: model-based final decision ----
    def _decide_heading(
        self,
        image: Image.Image,
        summary_heading: str,
        crop_heading: str,
    ) -> str:
        """
        Given candidate headings from:
          - full-page summary (summary_heading)
          - top-strip crop (crop_heading),

        ask the model to decide which one is the true main content heading,
        or output a different heading read directly from the page if both
        candidates are wrong.

        This is where we disambiguate main content heading vs. grand site title.
        """
        if not (summary_heading or crop_heading):
            return ""

        a = summary_heading or ""
        b = crop_heading or ""

        prompt = (
            "You are looking at a screenshot of a web page.\n"
            "Your task is to find the MAIN CONTENT HEADING of the page.\n\n"
            "Important:\n"
            "- The main content heading is the title of the main article or main content section.\n"
            "- Do NOT pick the site name, logo text, company name, browser tab title,\n"
            "  or generic banner slogan.\n"
            "- Focus on the heading that appears above the main body of content.\n\n"
            f"CANDIDATE_A: {a if a else '(none)'}\n"
            f"CANDIDATE_B: {b if b else '(none)'}\n\n"
            "Steps you should follow (do NOT write these steps out):\n"
            "1. Look carefully at the screenshot and locate the true main content heading.\n"
            "2. If CANDIDATE_A or CANDIDATE_B exactly match that heading, choose that candidate.\n"
            "3. If neither candidate is correct, read the correct main heading from the page and output it.\n\n"
            "You MUST respond in exactly ONE line:\n"
            "FINAL_HEADING: <exact heading text copied from the page>\n"
            "Do NOT add any explanation or extra text.\n"
        )

        text = self._llava_generate(prompt, image).strip()
        if not text:
            return clean_heading(a or b)

        # Take last non-empty line
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        last = lines[-1] if lines else text

        if "FINAL_HEADING" in last.upper():
            # Be tolerant to 'FINAL_HEADING:' or 'Final_Heading :'
            parts = last.split(":", 1)
            val = parts[1].strip() if len(parts) > 1 else ""
        else:
            val = last.strip()

        return clean_heading(val)

    # ---- main logic used by __call__ ----
    def generate(
        self,
        query: str,
        image: Image.Image,
        task_type: str = "",
        question: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Core entry point used by BaseAdapter.__call__.

        For HEADING_OCR_TASK + use_agent=True:
            - Use heading summary (full page) + top-strip crop.
            - Then let the model choose the final heading via _decide_heading.
        For WEBQA_TASK + use_agent=True:
            - Use VisualCoTAgent and take its final_answer.
            - Then run a refinement pass to trim the answer.
        Otherwise:
            - Fall back to simple one-shot generation.
        """
        try:
            # ---------- Heading OCR pipeline ----------
            if (
                task_type == HEADING_OCR_TASK
                and self.use_agent
                and self.agent is not None
            ):
                print("\n[LlavaHFAdapter][HEADING] Using heading OCR pipeline (summary + top strip + model decision)")

                # (a) Full-page heading summary
                raw_summary = self._heading_summary_call(image)
                print("[LlavaHFAdapter][HEADING] raw_summary:")
                print(raw_summary)

                summary_heading = _extract_top_heading_from_summary(raw_summary)
                print(f"[LlavaHFAdapter][HEADING] summary_heading: {summary_heading!r}")

                # (b) Top-strip crop (e.g. top 35% of the image)
                top_bbox = [0.0, 0.0, 1.0, 0.35]
                print(f"[LlavaHFAdapter][HEADING] top bbox: {top_bbox}")
                heading_view = crop_normalized(image, top_bbox)

                # use the provided heading_ocr_prompt as `query`
                raw_crop = self._llava_generate(query, heading_view)
                print("[LlavaHFAdapter][HEADING] raw_crop:")
                print(raw_crop)

                crop_heading = clean_heading(raw_crop)
                print(f"[LlavaHFAdapter][HEADING] crop_heading: {crop_heading!r}")

                if not (summary_heading or crop_heading):
                    print("[LlavaHFAdapter][HEADING] No candidates! Falling back to raw outputs.")
                    return crop_heading or summary_heading or raw_crop or raw_summary or ""

                # (c) Let the model decide which heading is correct
                final_heading = self._decide_heading(image, summary_heading, crop_heading)
                print(f"[LlavaHFAdapter][HEADING] final_heading: {final_heading!r}")
                return final_heading
            

            # ---------- WebQA multi-view pipeline + refiner ----------
            if (
                task_type == WEBQA_TASK
                and self.use_agent
                and self.agent is not None
            ):
                q = question if question is not None else query
                result = self.agent.run_chain(image, q)
                raw_final = result.get("final_answer", "") or ""
                print(f"[LlavaHFAdapter][WEBQA] raw_final: {raw_final!r}")

                refined = self._refine_answer(q, raw_final, image)
                print(f"[LlavaHFAdapter][WEBQA] refined_final: {refined!r}")

                return refined or raw_final
            
            if (
                task_type == ACTION_GROUND_TASK
                and self.use_agent
            ):
                q = question if question is not None else query
                return self._run_action_grounding(image, q)

            # ---------- Default: one-shot LLaVA for other tasks ----------
            return self._llava_generate(query, image)

        except Exception as e:
            print(f"[LlavaHFAdapter] Error during generation: {e}")
            return ""

    # ---- make adapter callable ----
    def __call__(
        self,
        query: str,
        image: Image.Image,
        task_type: str = "",
        **kwargs,
    ) -> str:
        """
        Allows using the adapter like:
            response = model_adapter(prompt, image, task_type=..., question=...)
        """
        return self.generate(query, image, task_type=task_type, **kwargs)
