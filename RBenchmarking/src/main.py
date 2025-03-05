import os 
import logging
import glob
from analyzer import Analyzer
from benchmark import RBenchmarking


if __name__ == "__main__":

    os.makedirs("../plots", exist_ok=True)
    os.makedirs("../logs", exist_ok=True)


    logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    level=logging.INFO,
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    filename="../logs/info.log",
    )

    logging.info("Application started successfully.")


    model_names = [
                   "convnext_large",
                   "efficientnet_v2_l",
                   "mobilenet_v3_large",
                   "alexnet", 
                   "vit_l_16",
                   "vit_b_32",
                   "swin_t", 
                   "wide_resnet101_2",
                   "resnext101_64x4d",
                   "shufflenet_v2_x2_0",
                   "swin_b", 
                   "swin_s", 
                   "swin_v2_t", 
                   "swin_v2_b", 
                   "swin_v2_s"
                   ]

    images_dir = "../Test Images"

    for filename in os.listdir(images_dir):
        folder_path = os.path.join(images_dir, filename)

        for model_name in model_names:
            logging.info(f"Running model {model_name}")

            rb = RBenchmarking(
                folder_path=folder_path,
                model_name=model_name,
            )

            aug_results = rb.compute_augmented_similarities_for_all_images()
            # Must be added here
            sorted_aug_results = sorted(
                aug_results.items(), key=lambda x: x[1], reverse=False
            )

            rb.plot(sorted_results=sorted_aug_results)

            sum_score = [res[1] for res in sorted_aug_results]

            average_score = sum(sum_score) / len(sum_score)
            rb._record_to_csv(similarity_scores=average_score, model_name=model_name)
            # break
        # break

    analyzer = Analyzer(paths="../plots")

    csv_dir = "../plots"
    csv_files = glob.glob(os.path.join(csv_dir, "**/*.csv"), recursive=True)

    analyzer = Analyzer(paths=csv_files)
    analyzer()