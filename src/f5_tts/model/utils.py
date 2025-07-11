from __future__ import annotations

import os
import random
from collections import defaultdict
from importlib.resources import files

import torch
from torch.nn.utils.rnn import pad_sequence

import jieba
from pypinyin import lazy_pinyin, Style


# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


# utils.py
import os
from importlib.resources import files

# utils.py
import os
from importlib.resources import files

# utils.py
import os
from importlib.resources import files
import logging # Optional: Use logging instead of print if preferred

# Setup basic logging if you want to use logger.info, logger.warning etc.
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# logger = logging.getLogger(__name__)

def get_tokenizer(path_or_dataset_name, tokenizer_type="pinyin"):
    """
    Loads tokenizer mapping and size. Handles space ' ' as a valid token.

    Args:
        path_or_dataset_name: EITHER the dataset name alias (e.g., "my_dataset")
                              OR the full path to a specific vocab.txt file.
        tokenizer_type: "pinyin", "char", or "custom". If "custom", path_or_dataset_name
                        MUST be the full path to the vocab.txt file.

    Returns:
        tuple: (vocab_char_map, vocab_size)
    """
    try: # Wrap core logic in try/except for better error context
        base_data_path = str(files("f5_tts").joinpath("../../data"))
    except Exception as e:
        print(f"[get_tokenizer] Warning: Could not determine base_data_path relative to package: {e}. Assuming current dir './data'.")
        base_data_path = "./data" # Fallback path

    vocab_char_map = {}
    vocab_size = 0
    vocab_file_path = None # The final path to the vocab.txt file

    print(f"[get_tokenizer] Received path/alias: '{path_or_dataset_name}', type: '{tokenizer_type}'") # Debug print

    if tokenizer_type == "custom":
        # Expect path_or_dataset_name to be the full path to the vocab.txt file
        if os.path.isfile(path_or_dataset_name):
             vocab_file_path = path_or_dataset_name
             print(f"[get_tokenizer] Custom type: Using provided file path directly: {vocab_file_path}")
        else:
             # Check if it's a directory containing vocab.txt (less likely for custom case now)
             potential_path = os.path.join(path_or_dataset_name, "vocab.txt")
             if os.path.isdir(path_or_dataset_name) and os.path.isfile(potential_path):
                  vocab_file_path = potential_path
                  print(f"[get_tokenizer] Custom type: Found vocab.txt inside provided directory: {vocab_file_path}")
             else:
                  raise FileNotFoundError(f"Custom tokenizer type specified, but the provided path is not a valid file or directory containing vocab.txt: '{path_or_dataset_name}'")

    elif tokenizer_type in ["pinyin", "char"]:
         # Expect path_or_dataset_name to be the dataset alias
         dataset_name_alias = path_or_dataset_name
         # Construct the expected directory path based on alias and type
         vocab_dir = os.path.join(base_data_path, f"{dataset_name_alias}_{tokenizer_type}")
         vocab_file_path = os.path.join(vocab_dir, "vocab.txt")
         print(f"[get_tokenizer] Default type: Checking primary path: {vocab_file_path}")

         if not os.path.isfile(vocab_file_path):
              print(f"[get_tokenizer] File not found at primary path. Checking fallbacks...")
              # Fallback 1: Directory without type suffix
              vocab_dir_no_suffix = os.path.join(base_data_path, dataset_name_alias)
              potential_path_no_suffix = os.path.join(vocab_dir_no_suffix, "vocab.txt")
              if os.path.isfile(potential_path_no_suffix):
                   vocab_file_path = potential_path_no_suffix
                   print(f"[get_tokenizer] Found vocab in directory without type suffix: {vocab_dir_no_suffix}")
              else:
                    # Fallback 2: Original Emilia ZH/EN (adjust if needed)
                    emilia_vocab_dir = os.path.join(base_data_path, f"Emilia_ZH_EN_{tokenizer_type}")
                    emilia_vocab_file = os.path.join(emilia_vocab_dir, "vocab.txt")
                    if os.path.isfile(emilia_vocab_file):
                         vocab_file_path = emilia_vocab_file
                         print(f"[get_tokenizer] Using fallback Emilia vocab: {vocab_file_path}")
                    else:
                         raise FileNotFoundError(f"Default vocab file not found for dataset '{dataset_name_alias}' and type '{tokenizer_type}' at expected primary path '{os.path.join(base_data_path, f'{dataset_name_alias}_{tokenizer_type}')}' or fallback paths.")
    else:
         raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    # --- Read the determined vocab file ---
    print(f"[get_tokenizer] Attempting to read final vocab path: {vocab_file_path}")
    line_count = 0
    processed_count = 0
    try:
        with open(vocab_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line_count += 1
                # --- MODIFIED LOGIC to Handle Space ---
                original_line_content = line.rstrip('\n\r') # Strip only line endings
                if i==0 and original_line_content==" ":
                    token_to_process = original_line_content # Strip leading/trailing whitespace
                else:
                    token_to_process = original_line_content.strip() # Strip leading/trailing whitespace

                '''
                # Determine if the original line was just a single space
                is_just_space = (char_stripped == '' and original_line_content == ' ')

                # Determine the token to actually process and add
                token_to_process = ' ' if is_just_space else char_stripped
                '''
                
                # Process if the stripped char has content OR if it was specifically just a space
                #if char_stripped or is_just_space:
                if token_to_process in vocab_char_map:
                     # Log duplicate but don't increment processed_count again
                     print(f"[get_tokenizer] Warning: Duplicate token '{token_to_process}' found at line {i+1}. Keeping first index {vocab_char_map[token_to_process]}.")
                else:
                     # Assign index based on processed order
                     vocab_char_map[token_to_process] = processed_count
                     processed_count += 1 # Increment count ONLY for new unique tokens
                # --- End of Modified Logic ---

        # Final size is the number of unique non-empty/space tokens processed
        vocab_size = processed_count
        print(f"[get_tokenizer] Read {line_count} lines, processed {processed_count} unique non-empty/space tokens.")

        # +++ Add Detailed Check +++
        print(f"[get_tokenizer] Final vocab_size variable = {vocab_size}")
        print(f"[get_tokenizer] Length of vocab_char_map keys = {len(vocab_char_map)}")
        # Check if counts match (they should now)
        if vocab_size != len(vocab_char_map):
            print(f"[get_tokenizer] !!! MISMATCH between processed_count ({vocab_size}) and map length ({len(vocab_char_map)}) !!!")
        # ++++++++++++++++++++++++++

        if vocab_size == 0:
             raise ValueError(f"Vocabulary file '{vocab_file_path}' resulted in zero processed tokens.")

    except FileNotFoundError: # Catch specific error
         print(f"[get_tokenizer] ERROR: Final determined vocab file path not found: {vocab_file_path}")
         raise # Re-raise the exception
    except Exception as e:
         print(f"Error reading vocabulary file {vocab_file_path}: {e}")
         raise # Re-raise other exceptions

    return vocab_char_map, vocab_size

def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False
