# agents/image_agent_llava_med.py

import os
import argparse
from llava.eval import model_vqa

def generate_image_captions_llava_med(model_path, image_folder, question_file, answers_file, conv_mode="mistral_instruct", temperature=0.0):
    print("\n🚀 Running LLaVA-Med VQA Inference...\n")

    # 准备 args
    args = argparse.Namespace(
        model_path=model_path,
        model_base=None,
        image_folder=image_folder,
        question_file=question_file,
        answers_file=answers_file,
        conv_mode=conv_mode,
        num_chunks=1,
        chunk_idx=0,
        temperature=temperature,
        top_p=None,
        num_beams=1
    )

    # 调用model_vqa的主函数
    model_vqa.eval_model(args)

    print("\n✅ LLaVA-Med VQA completed! Outputs saved at:", answers_file)
