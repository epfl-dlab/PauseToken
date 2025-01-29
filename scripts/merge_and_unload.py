import argparse
from peft import AutoPeftModelForCausalLM
import shutil
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and unload the model")
    parser.add_argument("--model_path", type=str, help="Path to the model to merge and unload")
    args = parser.parse_args()

    model = AutoPeftModelForCausalLM.from_pretrained(args.model_path, device_map="cpu")
    # Merge and unload the model
    print("merging and unloading the model...")
    model = model.merge_and_unload()
    print("done!")
    # Save the model
    tmp_dir = args.model_path + "_tmp"
    print("Saving merged model to temporary directory...")
    model.save_pretrained(tmp_dir)
    print(f"Model temporary saved in {tmp_dir}")

    print("Removing original model...")
    # remove original model
    shutil.rmtree(args.model_path)
    # rename the tmp directory
    os.rename(tmp_dir, args.model_path)
    print(f"Moved Merged Model back to {args.model_path}")
