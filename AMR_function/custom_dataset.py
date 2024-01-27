import torch
from torch.utils.data import Dataset
import os

PROMPT = f"Translate this AMR graph into natural language text:\n{{amr}}\n---\nTranslation:"

class AMRDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, split):
        self.load_data(dataset_path, split)
        self.tokenizer = tokenizer
    
    def load_data(self, dataset_path, split):
        graph_edge_path = os.path.join(dataset_path, split + ".graph.edge")
        graph_node_path = os.path.join(dataset_path, split + ".graph.node")
        graph_info_path = os.path.join(dataset_path, split + ".graph.info")
        sequence_source_path = os.path.join(dataset_path, split + ".sequence.source")
        sequence_target_path = os.path.join(dataset_path, split + ".sequence.target")
        
        self.graph_edge = []
        self.graph_node = []
        self.graph_info = []
        self.sequence_source = []
        self.sequence_target = []

        with open(sequence_source_path, "r") as f:
            self.sequence_source = f.read().split("\n")
            while self.sequence_source[-1] == '':
                self.sequence_source.pop()
        
        with open(sequence_target_path, "r") as f:
            self.sequence_target = f.read().split("\n")
            while self.sequence_target[-1] == '':
                self.sequence_target.pop()

        # with open(graph_edge_path, "r") as f:
        #     tmp_graph_edge = f.read().split("\n\n")
        #     tmp_graph_edge = [i.split("\n") for i in tmp_graph_edge]
        #     for i in range(len(tmp_graph_edge)):
        #         while len(tmp_graph_edge[i]) > 0 and tmp_graph_edge[i][0] == '':
        #             tmp_graph_edge[i] = tmp_graph_edge[i][1:]
        #     i = 0
        #     while i < len(tmp_graph_edge):
        #         if tmp_graph_edge[i] == []:
        #             tmp_graph_edge.pop(i)
        #         else:
        #             i += 1
        
        # with open(graph_node_path, "r") as f:
        #     self.graph_node = f.read().split("\n\n")
        #     self.graph_node = [i.split("\n") for i in self.graph_node]
        #     while self.graph_node[-1] == ['']:
        #         self.graph_node.pop()
        
        # with open(graph_info_path, "r") as f:
        #     tmp_graph_info = f.read().split("\n")
        #     tmp_graph_info = [i.split(" ") for i in tmp_graph_info]
        #     while tmp_graph_info[-1] == ['']:
        #         tmp_graph_info.pop()
        #     graph_edge_idx = 0
        #     graph_node_idx = 0
        #     for info in tmp_graph_info:
        #         edge_num = (len(info) - 2) // 2
        #         self.graph_info.append({
        #             "node_num": int(info[0]),
        #             "root": int(info[1]),
        #             "edge": [
        #                 [int(info[i]), int(info[i+1])]
        #                 for i in range(2, len(info), 2)
        #             ]
        #         })
        #         if edge_num == 0:
        #             self.graph_edge.append([])
        #         else:
        #             self.graph_edge.append(tmp_graph_edge[graph_edge_idx])
        #             graph_edge_idx += 1
        #         print(graph_node_idx)
        #         if len(self.graph_edge[-1]) != edge_num or len(self.graph_node[graph_node_idx]) != int(info[0]):
        #             # 删掉
        #             self.graph_edge.pop()
        #             self.graph_node.pop()
        #             self.graph_info.pop()
        #             self.sequence_source = self.sequence_source[:graph_node_idx] + self.sequence_source[graph_node_idx+1:]
        #             self.sequence_target = self.sequence_target[:graph_node_idx] + self.sequence_target[graph_node_idx+1:]
        #         graph_node_idx += 1
        

        # assert len(self.graph_edge) == len(self.graph_node) == len(self.graph_info) == len(self.sequence_source) == len(self.sequence_target)

        assert len(self.sequence_source) == len(self.sequence_target)
        self.length = len(self.graph_edge)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        # graph_edge = self.graph_edge[index]
        # graph_node = self.graph_node[index]
        # graph_info = self.graph_info[index]
        sequence_source = self.sequence_source[index]
        sequence_target = self.sequence_target[index]

        prompt = PROMPT.format(amr=sequence_source)
        prompt = self.tokenizer.encode(self.tokenizer.bos_token + prompt, add_special_tokens=False)
        answer = self.tokenizer.encode(sequence_target + self.tokenizer.eos_token, add_special_tokens=False)

        input_ids = prompt + answer
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt) + answer

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def get_item_without_answer(self, index):
        # graph_edge = self.graph_edge[index]
        # graph_node = self.graph_node[index]
        # graph_info = self.graph_info[index]
        sequence_source = self.sequence_source[index]
        sequence_target = self.sequence_target[index]

        prompt = PROMPT.format(amr=sequence_source)
        return prompt




def get_custom_dataset(dataset_config, tokenizer, split):
    return AMRDataset(tokenizer, dataset_config.data_path, split)
