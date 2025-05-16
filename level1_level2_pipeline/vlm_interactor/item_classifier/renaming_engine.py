import vlm_interactor
import json
import os
import glog
import sapien
import time

import vlm_interactor.vlm_interactor


class ItemClassifier:
    def __init__(self, model="GPT4o"):
        self.interactor = vlm_interactor.vlm_interactor.VLMInteractor(
            mode="online", model=model
        )
        self.interactor.initcount()
        self.interactor.chkcount()
        self.prompts = json.load(open("./vlm_interactor/prompts/renaming_engine.json"))

    def classify(self, img_path_folder, msg=None):

        image_files = [
            f
            for f in os.listdir(img_path_folder)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        new_name_dict = {}
        img_path_list = [os.path.join(img_path_folder, f) for f in image_files]

        for i in range(0, len(img_path_list), 1):
            self.interactor.add_content(
                content=self.prompts["task_config"]["system_prompt"],
                role="system",
                content_type="text",
            )
            self.interactor.add_content(
                content=self.prompts["task_config"]["user_prompt"],
                role="user",
                content_type="text",
            )
            self.interactor.add_content(
                content=f"previously you have classified these items: {list(new_name_dict.values())}, please DO NOT repeat them",
                role="user",
                content_type="text",
            )
            # glog.info(f'previously you have classified these items{new_name_dict.values()}, please do not repeat them')

            for j in range(i, min(i + 1, len(img_path_list))):
                self.interactor.add_content(
                    content=img_path_list[j], role="user", content_type="image"
                )
                self.interactor.add_content(
                    content=self.prompts["task_config"]["per_pic_prompt"],
                    role="user",
                    content_type="text",
                )
                status_code, answer1 = self.interactor.send_content_n_request()
                if (
                    status_code
                    == vlm_interactor.vlm_interactor.InteractStatusCode.SUCCESS
                ):

                    img_file = image_files[j]
                    print(answer1, img_file[: img_file.rfind(".")])
                    new_name_dict[img_file[: img_file.rfind(".")]] = answer1
                    self.interactor.add_content(
                        content=answer1, role="assistant", content_type="text"
                    )
                else:
                    glog.info("VLM Interactor send failed")
                    new_name_dict[img_file[: img_file.rfind(".")]] = (
                        "VLM Interactor send failed"
                    )
                pass

            self.interactor.clear_history()

        return new_name_dict


def main():
    # Example usage
    classifier = ItemClassifier()
    img_path_folder = "./image4classify"
    msg = "Classify these items"
    ts = time.perf_counter()
    new_names = classifier.classify(img_path_folder, msg)

    print(new_names)
    glog.info(f"Time taken: {time.perf_counter() - ts} seconds")


if __name__ == "__main__":
    main()
    # Example usage
