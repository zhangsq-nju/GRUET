import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class NLIScorer:
    def __init__(self, model_name="microsoft/deberta-large-mnli", device=None):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if self.device == "cuda":
            self.model = self.model.half().cuda()
        else:
            self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, a_list, b_list, judge="2entail", batch_size=256):

        if not a_list or not b_list:
            return []
        assert len(a_list) == len(b_list), "a_list and b_list must have the same length."

        all_results = []
        for i in range(0, len(a_list), batch_size):
            batch_a = a_list[i : i + batch_size]
            batch_b = b_list[i : i + batch_size]

            input_texts_ab = [f"{a} [SEP] {b}" for a, b in zip(batch_a, batch_b)]
            input_texts_ba = [f"{b} [SEP] {a}" for a, b in zip(batch_a, batch_b)]
            inputs_ab = self.tokenizer(
                input_texts_ab, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            inputs_ba = self.tokenizer(
                input_texts_ba, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.inference_mode():
                logits_ab = self.model(**inputs_ab).logits
                logits_ba = self.model(**inputs_ba).logits
            pred_ab = torch.argmax(logits_ab, dim=1)
            pred_ba = torch.argmax(logits_ba, dim=1)


            if judge == "1entail":
                batch_results = (pred_ab != 0) & (pred_ba != 0) & ((pred_ab == 2) | (pred_ba == 2))
            elif judge == "2entail":
                batch_results = (pred_ab == 2) & (pred_ba == 2)
            elif judge == "0contradict":
                batch_results = (pred_ab != 0) & (pred_ba != 0)
            else:
                raise ValueError(f"Unknown judge method: {judge}")

            all_results.extend(batch_results.tolist())

        return all_results
