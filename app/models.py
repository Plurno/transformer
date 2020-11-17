import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class Reader():
    def __init__(self, client, index):
        self.client = client
        self.index = index
        model_name = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=512)
        self.model = torch.jit.load("app/data/traced_roberta.pt")
#         self.model.to('cuda')
        
    def query(self, query, fields=['text'], filters=[], top_k=10, excluded_meta_data=[]):
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                            "query": query, 
                            "type": "most_fields", 
                            "fields": fields
                        }
                    }
                }
            }
        } 
        
        if filters:
            filter_clause = []
            for key, values in filters.items():
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            body["query"]["bool"]["filter"] = filter_clause

        if excluded_meta_data:
            body["_source"] = {"excludes": excluded_meta_data}

        result = self.client.search(index=self.index, body=body)["hits"]["hits"]
        self.contexts = [r['_source']['text'] for r in result]
        return self.contexts
    
    def infer(self, query):
        self.query(query)
        
        self.outputs = []
        self.answers = []
        for context in self.contexts:
            inputs = self.tokenizer(query, context, return_tensors="pt", add_special_tokens=True)
            input_ids = inputs["input_ids"].tolist()[0]

            # Get longest question to test out saved model. We can pad shorter answers. Probably better than trimming longer ones
            l = len(input_ids)
            if l >= 512:
                continue

            outputs = self.model(**inputs)
            self.outputs.append(outputs)
            
            answer = {}
            answer_start_scores = outputs[0]
            answer_end_scores = outputs[1]
            answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            response = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            
            answer['score'] = max([answer_start_scores[0][answer_start].item(), answer_end_scores[0][answer_end].item()])
            answer['answer'] = response
            self.answers.append(answer)

        self.answers.sort(key=lambda x: x['score'], reverse = True)   
        return self.answers