# pathchecker = pathchecker.PathChecker()
# # 使用示例
# json_path = f'..out_data/datasets_to_json/short_flickr30k_images_datasets.json'
# image_folder = '..dataset/short_flickr30k_images_datasets/short_flickr30k_images_datasets'
# output_path = '..out_data/img_json_hdf5'
#
# if pathchecker.check_path_exists(json_path):
#     json_path = pathchecker.process_path(json_path)
# else:
#     print("json not exists")
# image_folder = pathchecker.check_path_exists(image_folder)
# output_path = pathchecker.check_path_exists(output_path, True)
# create_input_files('flickr30k', json_path, image_folder, 5, 5, output_path)