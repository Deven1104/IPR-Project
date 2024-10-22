# /data/multimodal-garment-designer/fashion_data

python src/eval.py --dataset_path /data/multimodal-garment-designer/fashion_data --batch_size 32 --mixed_precision fp16 --output_dir ./output --save_name save_name_tbd --num_workers_test 2 --sketch_cond_rate 0.2 --dataset vitonhd --start_cond_rate 0.0 --test_order paired
python src/eval.py --dataset_path /data/multimodal-garment-designer/fashion_data --batch_size 32 --mixed_precision fp16 --output_dir ./output --save_name save_name_tbd --num_workers_test 2 --sketch_cond_rate 0.2 --dataset vitonhd --start_cond_rate 0.0 --test_order unpaired