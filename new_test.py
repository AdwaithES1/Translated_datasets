import json
import torch
import os
import shutil
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from sacrebleu.metrics import CHRF

# This dictionary remains the same
SUPPORTED_LANGUAGES = {
    1: ("Assamese", "asm_Beng"),
    2: ("Bengali", "ben_Beng"),
    3: ("Bodo", "brx_Deva"),
    4: ("Dogri", "dgo_Deva"),
    5: ("Gujarati", "guj_Gujr"),
    6: ("Hindi", "hin_Deva"),
    7: ("Kannada", "kan_Knda"),
    8: ("Kashmiri (Arabic)", "kas_Arab"),
    9: ("Konkani", "kok_Deva"),
    11: ("Maithili", "mai_Deva"),
    11: ("Malayalam", "mal_Mlym"),
    12: ("Manipuri", "mni_Beng"),
    13: ("Marathi", "mar_Deva"),
    14: ("Nepali", "npi_Deva"),
    15: ("Oriya", "ory_Orya"),
    16: ("Punjabi", "pan_Guru"),
    17: ("Sanskrit", "san_Deva"),
    18: ("Santali", "sat_Olck"),
    19: ("Sindhi", "snd_Arab"),
    20: ("Tamil", "tam_Taml"),
    21: ("Telugu", "tel_Telu"),
    22: ("Urdu", "urd_Arab")
}

class IndicTranslator:
    """A wrapper class for the IndicTrans2 model."""
    def __init__(self, model_name, src_lang, tgt_lang, batch_size=32):
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_name} onto {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="flash_attention_2" if self.device == "cuda" else None
        ).to(self.device)
        self.processor = IndicProcessor(inference=True)

    def translate_texts(self, texts):
        # Filter out empty or whitespace-only strings to avoid sending them to the model
        non_empty_texts = [text for text in texts if text and text.strip()]
        if not non_empty_texts:
            return [""] * len(texts)

        # Keep track of original positions to place translations back correctly
        original_indices = {i: text for i, text in enumerate(texts) if text and text.strip()}
        texts_to_translate = list(original_indices.values())

        batch = self.processor.preprocess_batch(texts_to_translate, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        inputs = self.tokenizer(
            batch, truncation=True, padding="longest", return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, use_cache=True, min_length=0, max_length=512, num_beams=5, num_return_sequences=1
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        postprocessed = self.processor.postprocess_batch(decoded, lang=self.tgt_lang)

        # Place translations back into a list of the original size
        final_translations = [""] * len(texts)
        for i, original_index in enumerate(original_indices.keys()):
            final_translations[original_index] = postprocessed[i]

        return final_translations


def run_translation_pipeline(input_file, output_file, dataset_features):
    """
    Core translation engine. Prompts for columns and runs the translation/scoring.
    """
    print(f"Loading dataset from file: '{input_file}' ...")
    try:
        dataset = load_dataset('json', data_files=input_file, split='train')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # --- Interactive Prompts ---
    text_columns = prompt_for_columns_to_translate(list(dataset.features.keys()), dataset_features)
    if not text_columns:
        print("No columns selected for translation. Aborting.")
        return

    tgt_lang = "mal_Mlym" # Hardcoded for Malayalam
    print(f"\nTarget language set to: Malayalam ({tgt_lang})")
    batch_size = prompt_for_batch_size()
    chrf_threshold = prompt_for_chrf_threshold()

    # --- Model Initialization & Processing Loop ---
    chrf = CHRF(word_order=2) # CHRF++
    en_to_indic_translator = IndicTranslator("ai4bharat/indictrans2-en-indic-1B", "eng_Latn", tgt_lang, batch_size)
    indic_to_en_translator = IndicTranslator("ai4bharat/indictrans2-indic-en-1B", tgt_lang, "eng_Latn", batch_size)
    saved_count = 0
    with open(output_file, "w", encoding="utf-8") as f, tqdm(range(0, len(dataset), batch_size)) as pbar:
        for i in pbar:
            pbar.set_description(f"Translated & Scored | Saved: {saved_count}")
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            batch_dict = batch.to_dict()
            batch_size_actual = len(batch)

            texts_dict = {col: [str(batch_dict[col][idx] or "") for idx in range(batch_size_actual)] for col in text_columns}
            translations_dict = {}
            scores_dict = {}

            for col in text_columns:
                original_texts = texts_dict[col]
                indic_translations = [""] * len(original_texts)
                scores = [0.0] * len(original_texts)
                try:
                    translated_batch = en_to_indic_translator.translate_texts(original_texts)
                    back_translated_batch = indic_to_en_translator.translate_texts(translated_batch)
                    for idx, (original, back_translated) in enumerate(zip(original_texts, back_translated_batch)):
                        if original and original.strip():
                            indic_translations[idx] = translated_batch[idx]
                            scores[idx] = chrf.sentence_score(back_translated, [original]).score
                except Exception as e:
                    print(f"\nBatch processing error: {e}")

                translations_dict[col] = indic_translations
                scores_dict[col] = scores

            for idx in range(batch_size_actual):
                should_save = all(scores_dict[col][idx] >= chrf_threshold for col in text_columns)

                if should_save:
                    saved_count += 1
                    # Create a new item, keeping only non-translated fields from the original
                    original_item = {col: batch_dict[col][idx] for col in batch_dict}
                    new_item = {k: v for k, v in original_item.items() if k not in text_columns}

                    # Add the translated fields back, but using the original column names
                    # This replaces the English text with the Malayalam translation.
                    for col in text_columns:
                        new_item[col] = translations_dict[col][idx]

                    f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                    f.flush()

    print(f"\nProcessing complete. Saved {saved_count} of {len(dataset)} records to {output_file}")


# ----- HELPER PROMPT FUNCTIONS -----
def prompt_for_columns_to_translate(columns, features):
    """
    Prompts the user to select one or more columns to translate.
    Returns a list of column names.
    """
    print("\nAvailable columns in the file:")
    for i, col in enumerate(columns):
        print(f"  {i+1}. {col}")
    print("\nWhich columns would you like to translate?")
    print("Enter numbers separated by commas (e.g., 1, 3), or 'ALL' for all text-based columns.")

    while True:
        choice_str = input("Columns to translate: ").strip()
        if choice_str.lower() == 'all':
            return [col for col in columns if features[col].dtype in ["string", "list"]]

        selected_columns = []
        try:
            choices = [int(c.strip()) for c in choice_str.split(',')]
            valid_choices = True
            for choice in choices:
                if 1 <= choice <= len(columns):
                    selected_columns.append(columns[choice - 1])
                else:
                    print(f"Error: '{choice}' is not a valid number. Please choose from 1 to {len(columns)}.")
                    valid_choices = False
                    break
            if valid_choices:
                return list(dict.fromkeys(selected_columns)) # Remove duplicates and preserve order
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas, or 'ALL'.")

def prompt_for_batch_size():
    batch_size = input("Enter batch size (default: 32): ").strip()
    return int(batch_size) if batch_size.isdigit() else 32

def prompt_for_chrf_threshold():
    while True:
        threshold_str = input("Enter CHRF++ threshold [0-100] (default: 0.0): ").strip()
        if not threshold_str: return 0.0
        try:
            threshold = float(threshold_str)
            if 0.0 <= threshold <= 100.0: return threshold
            else: print("Value must be between 0.0 and 100.0.")
        except ValueError: print("Invalid number.")


# ----- MAIN ORCHESTRATOR -----
def main():
    """
    Orchestrates the selective translation of a BeIR dataset.
    """
    # 1. --- Setup Paths ---
    dataset_name = input("Enter a name for your output BeIR dataset folder: ").strip()
    os.makedirs(dataset_name, exist_ok=True)

    corpus_input = input("Enter the path to the original corpus file (e.g., path/to/corpus.jsonl): ").strip()
    queries_input = input("Enter the path to the original queries file (e.g., path/to/queries.jsonl): ").strip()
    qrels_input_dir = input("Enter the path to the original qrels directory: ").strip()

    # Define output paths
    corpus_output = os.path.join(dataset_name, "corpus.jsonl")
    queries_output = os.path.join(dataset_name, "queries.jsonl")
    qrels_output_dir = os.path.join(dataset_name, "qrels")

    # Get features to help with column selection later
    try:
        corpus_features = load_dataset('json', data_files=corpus_input, split='train').features if os.path.exists(corpus_input) else None
        queries_features = load_dataset('json', data_files=queries_input, split='train').features if os.path.exists(queries_input) else None
    except Exception as e:
        print(f"Could not pre-load dataset features. Error: {e}")
        corpus_features, queries_features = {}, {}


    # 2. --- Process Corpus ---
    choice = input("\nDo you want to TRANSLATE the corpus file? (y/n): ").strip().lower()
    if choice == 'y':
        print("\n--- Preparing to Translate Corpus ---")
        if os.path.exists(corpus_input) and corpus_features:
            run_translation_pipeline(corpus_input, corpus_output, corpus_features)
        else:
            print(f"ERROR: Input file not found or features could not be read from {corpus_input}. Cannot translate.")
    else:
        print("\n--- Copying Corpus ---")
        if os.path.exists(corpus_input):
            shutil.copy(corpus_input, corpus_output)
            print(f"Successfully copied original corpus to {corpus_output}")
        else:
            print(f"ERROR: Input file not found at {corpus_input}. Cannot copy.")

    # 3. --- Process Queries ---
    choice = input("\nDo you want to TRANSLATE the queries file? (y/n): ").strip().lower()
    if choice == 'y':
        print("\n--- Preparing to Translate Queries ---")
        if os.path.exists(queries_input) and queries_features:
            run_translation_pipeline(queries_input, queries_output, queries_features)
        else:
            print(f"ERROR: Input file not found or features could not be read from {queries_input}. Cannot translate.")
    else:
        print("\n--- Copying Queries ---")
        if os.path.exists(queries_input):
            shutil.copy(queries_input, queries_output)
            print(f"Successfully copied original queries to {queries_output}")
        else:
            print(f"ERROR: Input file not found at {queries_input}. Cannot copy.")

    # 4. --- Process Qrels (Always Copy) ---
    print("\n--- Copying Qrels Directory ---")
    if os.path.isdir(qrels_input_dir):
        if os.path.exists(qrels_output_dir):
            shutil.rmtree(qrels_output_dir) # Remove old dir if it exists before copying
        shutil.copytree(qrels_input_dir, qrels_output_dir)
        print(f"Successfully copied qrels to {qrels_output_dir}")
    else:
        print(f"ERROR: Input directory not found at {qrels_input_dir}. Cannot copy.")

    print(f"\n✅ BeIR dataset processing complete. Your new dataset is ready in the '{dataset_name}' folder.")


if __name__ == "__main__":
    main()
