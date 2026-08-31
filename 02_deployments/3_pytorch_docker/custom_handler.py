import torch
from ts.torch_handler.base_handler import BaseHandler

class CustomHandler(BaseHandler):
    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        model_path = f"{model_dir}/model.pt"

        self.model = torch.jit.load(model_path)
        self.model.eval()

    def preprocess(self, data):
        row = data[0]
        body = row.get("body", {})
        data = body.get("data", {})
        tensor = torch.tensor(data, dtype=torch.float32)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def inference(self, inputs):
        with torch.no_grad():
            return self.model(inputs)

    def postprocess(self, outputs):
        return outputs.tolist()
