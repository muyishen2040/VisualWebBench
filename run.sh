model_name=blip2_opt_2_7b #qwen_vl_7b_hf #llava_7b
task_type=heading_ocr,webqa #action_ground #web_caption,webqa,heading_ocr,element_ocr,element_ground,action_prediction,action_ground
export HF_USE_FLASH_ATTENTION=0

python $DEBUG_MODE run.py \
    --model_name $model_name \
    --dataset_name_or_path webbench/WebBench \
    --task_type $task_type \
    --gpus 0

