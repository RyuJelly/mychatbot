from config.DatabaseConfig import *
from utils.Database import Database
from utils.Preprocess import Preprocess

# 전처리 객체 생성
p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict_v2.bin',
               userdic='../utils/user_dic.tsv')

# 질문/답변 학습 디비 연결 객체 생성
db = Database(
    host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME
)
db.connect()    # 디비 연결

query = "사이다 몇개 남아있어?"

# 의도 파악
from models.intent.IntentModel import IntentModel
intent = IntentModel(model_name='../models/intent/intent_model_v2.h5', proprocess=p)
predict = intent.predict_class(query)
intent_name = intent.labels[predict]

# 개체명 인식
# from models.ner.NerModel import NerModel
# ner = NerModel(model_name='../models/ner/ner_model.h5', proprocess=p)
# predicts = ner.predict(query)
# ner_tags = ner.predict_tags(query)

print("질문 : ", query)
print("=" * 100)
print("의도 파악 : ", intent_name)
# print("개체명 인식 : ", predicts)
# print("답변 검색에 필요한 NER 태그 : ", ner_tags)
print("=" * 100)

# 답변 검색
from utils.FindAnswer import FindAnswer


f = FindAnswer(db)
answer_text = f.search(intent_name)
answer = f.tag_to_word(answer_text)

# except:
#     answer = "죄송해요 무슨 말인지 모르겠어요"

print("답변 : ", answer)

db.close() # 디비 연결 끊음