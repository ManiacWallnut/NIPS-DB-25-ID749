from vlm_interactor.VLMEvalKit.vlmeval.smp import *
from vlm_interactor.VLMEvalKit.vlmeval.api.base import BaseAPI
from time import sleep
import base64
import mimetypes
from PIL import Image

alles_url = "https://openxlab.org.cn/gw/alles-apin-hub/v1/claude/v1/text/chat"
alles_headers = {"alles-apin-token": "", "Content-Type": "application/json"}
# official_url = 'https://api.anthropic.com/v1/messages'
# official_headers = {
#     'x-api-key': '',
#     'anthropic-version': '2023-06-01',
#     'content-type': 'application/json'
# }
# official_url = 'https://platform.llmprovider.ai/v1/messages'
# official_headers = {
#     'x-api-key': '',
#     'anthropic-version': '2023-06-01',
#     'Content-Type': 'application/json'
# }


official_url = "https://openrouter.ai/api/v1/chat/completions"
official_headers = {
    "Authorization": "Bearer ",
    "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
}


class Claude_Wrapper(BaseAPI):

    is_api: bool = True

    def __init__(
        self,
        backend: str = "alles",
        model: str = "claude-3-opus-20240229",
        key: str = None,
        retry: int = 10,
        wait: int = 3,
        system_prompt: str = None,
        verbose: bool = False,
        temperature: float = 0,
        max_tokens: int = 1024,
        **kwargs,
    ):

        if os.environ.get("ANTHROPIC_BACKEND", "") == "official":
            backend = "official"

        assert backend in ["alles", "official"], f"Invalid backend: {backend}"
        self.backend = backend
        self.url = alles_url if backend == "alles" else official_url
        self.model = "anthropic/claude-3-opus"  # model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.headers = alles_headers if backend == "alles" else official_headers

        if key is not None:
            self.key = key
        else:
            self.key = (
                os.environ.get("ALLES", "")
                if self.backend == "alles"
                else os.environ.get("ANTHROPIC_API_KEY", "")
            )  # noqa: E501

        # if self.backend == 'alles':
        #     self.headers['alles-apin-token'] = self.key
        # else:
        #     self.headers['x-api-key'] = self.key

        self.headers = official_headers

        super().__init__(
            retry=retry,
            wait=wait,
            verbose=verbose,
            system_prompt=system_prompt,
            **kwargs,
        )

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x["type"] == "image" for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg["type"] == "text" and msg["value"] != "":
                    content_list.append(dict(type="text", text=msg["value"]))
                elif msg["type"] == "image":
                    pth = msg["value"]
                    suffix = osp.splitext(pth)[-1].lower()
                    media_type = mimetypes.types_map.get(suffix, None)
                    assert media_type is not None

                    from PIL import Image

                    img = Image.open(msg["value"])

                    b64 = encode_image_to_base64(img, target_size=512)
                    content_list.append(
                        dict(
                            type="image_url",
                            image_url=dict(
                                url=f"data:image/jpeg;base64,{b64}", detail="low"
                            ),
                            # source={
                            #     'type': 'base64',
                            #     'media_type': media_type,
                            #     'data': encode_image_file_to_base64(pth, target_size=4096)
                            # }
                        )
                    )
        else:
            assert all([x["type"] == "text" for x in inputs])
            text = "\n".join([x["value"] for x in inputs])
            content_list = [dict(type="text", text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(["type" in x for x in inputs]) or np.all(
            ["role" in x for x in inputs]
        ), inputs
        if "role" in inputs[0]:
            assert inputs[-1]["role"] == "user", inputs[-1]
            for item in inputs:
                input_msgs.append(
                    dict(
                        role=item["role"], content=self.prepare_itlist(item["content"])
                    )
                )
        else:
            input_msgs.append(dict(role="user", content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.prepare_inputs(inputs),
            **kwargs,
        }
        if self.system_prompt is not None:
            payload["system"] = self.system_prompt

        # import ipdb; ipdb.set_trace()
        response = requests.post(
            self.url, headers=self.headers, data=json.dumps(payload)
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg

        # print(f"Request payload: {json.dumps(payload, indent=2)}")
        print(f"API Response: {response.text}")
        # import ipdb; ipdb.set_trace()

        try:
            resp_struct = json.loads(response.text)
            # answer = resp_struct['data']['content'][0]['text'].strip()
            answer = resp_struct["content"][0]["text"].strip()
        except Exception as err:
            if self.verbose:
                self.logger.error(f"{type(err)}: {err}")
                self.logger.error(
                    response.text if hasattr(response, "text") else response
                )

        return ret_code, answer, response


class Claude3V(Claude_Wrapper):

    def generate(self, message, dataset=None):
        return super(Claude_Wrapper, self).generate(message)
