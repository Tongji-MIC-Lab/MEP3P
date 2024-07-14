from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import masked_softmax,  replace_masked_values
from allennlp.nn import InitializerApplicator
from models.transformer import EncoderLayer
from models.transformer_depth_fusion import DepthEncoderLayer
import torch.nn as nn

@Model.register("Depth_blip_model")

class Depth_fusion_model(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
                 class_embs: bool = True,
                 reasoning_use_obj: bool = True,
                 reasoning_use_answer: bool = True,
                 reasoning_use_question: bool = True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(Depth_fusion_model, self).__init__(vocab)

        #self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None
        self.span_encoder = TimeDistributed(span_encoder)

        self.obj_downsample = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True),
        )

        #self.fc_3d = torch.nn.Linear(6, 512)
        self.boxes_fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(512 + 6 + 512, 512),
            torch.nn.ReLU(inplace=True),
        )

        self.depth_imgs_convs = torch.nn.Sequential(
            torch.nn.Dropout(p=0.1),
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            ## [16,32,32]
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            ## [32,16,16]
            torch.nn.Conv2d(32, 16, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            ## [16,8,8]
        )
        self.depth_imgs_fc = torch.nn.Sequential(
            torch.nn.Linear(16*8*8, 512),
            torch.nn.ReLU(inplace=True),
            )
        
        self.qa_input_dropout = nn.Dropout(0.1)
        self.qa_encoder1 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8)
        self.qa_encoder2 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8)
        self.qa_encoder3 = EncoderLayer(512, 512, 512, 0.1, 0.1, 8)
        
        self.va_input_dropout = nn.Dropout(0.1)
        self.va_encoder1 = DepthEncoderLayer(512, 512, 512, 0.1, 0.1, 8)
        self.va_encoder2 = DepthEncoderLayer(512, 512, 512, 0.1, 0.1, 8)
        self.va_encoder3 = DepthEncoderLayer(512, 512, 512, 0.1, 0.1, 8)

        self.reasoning_input_dropout = nn.Dropout(0.1)
        self.reasoning_encoder1 = EncoderLayer(512*3, 512*3, 512*3, 0.1, 0.1, 8)
        self.reasoning_encoder2 = EncoderLayer(512*3, 512*3, 512*3, 0.1, 0.1, 8) 
        
        ##=======================useless
        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question
        ##========================
        
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(1536+768, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )
        
        self.score_fc = nn.Linear(1536, 1536)
        self.blip_score_fc = nn.Linear(768, 768)

        self._accuracy = CategoricalAccuracy()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.b_cls_loss = torch.nn.BCELoss()

        self.att_dict = {}

        initializer(self)

    def _collect_obj_reps(self, span_tags, object_reps):

        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):

        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)
        span_rep = torch.cat((span['bert'], retrieved_feats), -1)  # bs,4,q_length(13),512+768;
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)
        return self.span_encoder(span_rep, span_mask), retrieved_feats

    def forward(self,
                det_features:torch.Tensor,
                boxes: torch.Tensor,
                features_3d:torch.Tensor,
                crop_depth_img:torch.Tensor,
                blip_img:torch.Tensor,
                blip_mm:torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        max_len = int(box_mask.sum(1).max().item())
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        det_features = det_features[:,:max_len,:]

        features_3d = features_3d[:, :max_len]
        crop_depth_img = crop_depth_img[:, :max_len, :, :]

        obj_reps = det_features
        obj_reps = self.obj_downsample(obj_reps)

        #features_3d = self.fc_3d(features_3d)
        b_shape = crop_depth_img.shape[0]
        l_shape = crop_depth_img.shape[1]
        img_shape = crop_depth_img.shape[-1]
        crop_depth_img = crop_depth_img.reshape(b_shape*l_shape, img_shape, img_shape).unsqueeze(1)
        crop_depth_img = self.depth_imgs_convs(crop_depth_img).reshape(b_shape*l_shape, 16*8*8)
        crop_depth_img = self.depth_imgs_fc(crop_depth_img).reshape(b_shape, l_shape, -1)

        obj_reps = torch.cat([obj_reps, features_3d, crop_depth_img], dim=-1)
        obj_reps = self.boxes_fc(obj_reps)

        # Now get the question representations
        q_rep_ori, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps)  #
        a_rep_ori, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps)  #

        o_rep = obj_reps ## [b,max_len,512]

        ## [b,4,len]
        max_answer_len = int(answer_mask.sum(dim=2).max().item())
        max_question_len = int(question_mask.sum(dim=2).max().item())

        q_rep = q_rep_ori[:,:,:max_question_len,:]
        a_rep = a_rep_ori[:,:,:max_answer_len,:]
        question_mask = question_mask[:,:,:max_question_len]
        answer_mask = answer_mask[:,:,:max_answer_len]
        
        ## depth values of boxes->[b, len]
        depth_value = features_3d[:,:,2]
        ## 3d coord of boxes->[b, len, 3]
        coord_3d = features_3d[:,:,:3]
        # ## mean depth values of boxes for words->[b]
        # mean_depth_value = torch.mean(depth_value[:,:-1], dim=-1)

        ## VA encoder ===================================================================================================================
        va_v_rep = o_rep.unsqueeze(1).repeat(1,4,1,1)
        va_v_rep = va_v_rep.view(va_v_rep.shape[0] * va_v_rep.shape[1], va_v_rep.shape[2], va_v_rep.shape[3])
        va_a_rep = a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3])
        va_nodes = torch.cat([va_v_rep, va_a_rep],-2) ## [b*4,o_l+a_l, 512]
        
        va_v_mask = box_mask.unsqueeze(1).repeat(1,4,1)
        va_v_mask = va_v_mask.view(va_v_mask.shape[0] * va_v_mask.shape[1], va_v_mask.shape[2], 1)
        va_a_mask = answer_mask.view(answer_mask.shape[0] * answer_mask.shape[1], answer_mask.shape[2], 1)
        va_mask = torch.cat([va_v_mask, va_a_mask],-2)

        va_nodes1 = self.va_input_dropout(va_nodes)

        ## va encoder
        va_output1, b1 ,va_A1, mean_depth_value_1 = self.va_encoder1(va_nodes1, va_mask, depth_value, coord_3d, va_v_mask, va_a_mask)

        self.att_dict["va_1"] = b1

        masked_va_output1 = va_output1*va_mask
        va_nodes2 = masked_va_output1 

        va_output2, b2 , va_A2, mean_depth_value_2 = self.va_encoder2(va_nodes2, va_mask, depth_value, coord_3d, va_v_mask, va_a_mask, mean_depth_value_1)

        self.att_dict["va_2"] = b2

        masked_va_output2 = va_output2*va_mask  
        va_nodes3 = masked_va_output2 

        va_output3, b3 , va_A3, mean_depth_value_3 = self.va_encoder3(va_nodes3, va_mask, depth_value, coord_3d, va_v_mask, va_a_mask, mean_depth_value_2)

        self.att_dict["va_3"] = b3

        va_output = va_output3*va_mask

        va_output = va_output[:,o_rep.shape[1]:,:].view(a_rep.shape[0], a_rep.shape[1], a_rep.shape[2], 512)

        ## QA encoder===============================================================================================================
        qa_q_rep = q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3])
        qa_a_rep = a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3])
        qa_nodes = torch.cat([qa_q_rep, qa_a_rep],-2) ## [b*4,q_l+a_l, 512]
     
        qa_q_mask = question_mask.view(question_mask.shape[0] * question_mask.shape[1], question_mask.shape[2], 1)
        qa_a_mask = answer_mask.view(answer_mask.shape[0] * answer_mask.shape[1], answer_mask.shape[2], 1)
        qa_mask = torch.cat([qa_q_mask, qa_a_mask],-2)

        qa_nodes1 = self.qa_input_dropout(qa_nodes)

        ## qa encoder
        qa_output1, _ , qa_A1 = self.qa_encoder1(qa_nodes1, qa_mask)

        #self.att_dict["qa_1"] = qa_A1

        masked_qa_output1 = qa_output1*qa_mask
        qa_nodes2 = masked_qa_output1

        qa_output2, _ , qa_A2 = self.qa_encoder2(qa_nodes2, qa_mask)

        #self.att_dict["qa_2"] = qa_A2

        masked_qa_output2 = qa_output2*qa_mask  
        qa_nodes3 = masked_qa_output2 

        qa_output3, _ , qa_A3 = self.qa_encoder3(qa_nodes3, qa_mask)

        #self.att_dict["qa_3"] = qa_A3

        qa_output = qa_output3*qa_mask
        
        qa_output = qa_output[:,q_rep.shape[2]:,:].view(a_rep.shape[0], a_rep.shape[1], a_rep.shape[2], 512)
        
        ## reasoning =====================================================================================================================
        reasoning_inp = torch.cat([a_rep, va_output, qa_output], -1)

        reasoning_output = reasoning_inp.view(a_rep.shape[0]*a_rep.shape[1],a_rep.shape[2],1536)
        
        reasoning_nodes = reasoning_output
  
        reasoning_nodes1 = self.reasoning_input_dropout(reasoning_nodes)

        reasoning_output1, _ , reasoning_A1 = self.reasoning_encoder1(reasoning_nodes1, qa_a_mask)
        
        #self.att_dict["reason_1"] = reasoning_A1

        masked_reasoning_output1 = reasoning_output1*qa_a_mask
        reasoning_nodes2 = masked_reasoning_output1

        reasoning_output2, _ , reasoning_A2 = self.reasoning_encoder2(reasoning_nodes2, qa_a_mask)
        
        #self.att_dict["reason_2"] = reasoning_A2

        reasoning_a_rep = reasoning_output2
        
        # ====================
        masked_pool_rep = reasoning_a_rep*qa_a_mask.float()

        score_rep = self.score_fc(masked_pool_rep)
        score_rep = masked_softmax(score_rep, qa_a_mask, dim=-2)
        
        pool_rep = torch.sum(score_rep*masked_pool_rep, dim=-2).view(a_rep.shape[0], a_rep.shape[1], 1536)

        # ====================
        blip_score_rep = self.blip_score_fc(blip_mm)
        blip_score_rep = F.softmax(blip_score_rep, dim=-2)
        
        blip_pool_rep = torch.sum(blip_score_rep*blip_mm, dim=-2).view(blip_mm.shape[0], blip_mm.shape[1], 768)

        pool_rep = torch.concat([pool_rep, blip_pool_rep], dim=-1)
        
        logits = self.final_mlp(pool_rep).squeeze(2)  
        ###########################################

        class_probabilities = F.softmax(logits, dim=-1)
        option_score = torch.sigmoid(logits)

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                        "att_dict": self.att_dict
                       }
        if label is not None:
            label_one_hot = label.unsqueeze(-1)
            bce_label = torch.zeros(label.shape[0], 4).cuda()
            bce_label.scatter_(1,label_one_hot,1).cuda()

            option_score = option_score.view(-1)
            bce_label = bce_label.view(-1)

            loss_cls = self.cls_loss(logits, label.long().view(-1))
            loss_b_cls = self.b_cls_loss(option_score, bce_label)
            self._accuracy(logits, label)
            output_dict["loss"] = loss_cls+loss_b_cls
            output_dict["cls_loss"] = loss_cls
            output_dict["b_cls_loss"] = loss_b_cls
            output_dict["label"] = label.long().view(-1)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
