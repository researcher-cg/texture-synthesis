import helper
import tf_helper

input_w = int(input("Enter width (ORIGINAL IMAGE): "))   # width of input image(original image will be scaled down to this width), width of generated image
input_h = int(input("Enter height (ORIGINAL IMAGE): "))   # height of input image(original image will be scaled down to this height), height of generated image
filename = input("Filename (ORIGINAL IMAGE): ")
input_file = "./image_resources/original/" + filename +  ".jpg"
output_path = "./image_resources/processed/"
output_file = filename+"_processed.jpg"

texture_array = helper.resize_and_rescale_img(input_file, input_w, input_h, output_path, output_file)
filename_normal = "Stylized_Cliff_Rock512Normals"
normal_array = helper.resize_and_rescale_img("./image_resources/original/" + filename_normal +  ".jpg", input_w, input_h, output_path, output_file)

result_filename = filename + "-Normals.jpg"
final = helper.post_process_and_display(texture_array, "./image_resources/outputs/", result_filename, normal_array, save_file=True)
