import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing


# 의도 분류 모델 모듈
class IntentModel:
    def __init__(self, model_name, proprocess):

        # 의도 클래스 별 레이블
        self.labels = {0: "인사", 1: "레시피저장", 2: "레시피추천", 3: "현재식재료", 4: "달걀개수", 5: "레몬개수", 6: "자두개수", 7: "오이개수", 8: "사이다개수", 9: "당근개수",
                       10: "애호박개수", 11: "파인애플개수", 12: "사과개수", 13: "양파개수", 14: "마늘개수", 15: "토마토개수", 16: "브로콜리개수", 17: "깻잎개수",
                       18: "가지개수", 19: "단호박개수", 20: "무개수", 21: "양배추개수", 22: "파프리카개수", 23: "야쿠르트개수", 24: "맥주개수", 25: "콜라개수", 26: "옥수수개수"}

        # 의도 분류 모델 불러오기
        self.model = load_model(model_name)

        # 챗봇 Preprocess 객체
        self.p = proprocess


    # 의도 클래스 예측
    def predict_class(self, query):
        # 형태소 분석
        pos = self.p.pos(query)

        # 문장내 키워드 추출(불용어 제거)
        keywords = self.p.get_keywords(pos, without_tag=True)
        sequences = [self.p.get_wordidx_sequence(keywords)]

        # 단어 시퀀스 벡터 크기
        from config.GlobalParams import MAX_SEQ_LEN

        # 패딩처리
        padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

        predict = self.model.predict(padded_seqs)
        predict_class = tf.math.argmax(predict, axis=1)
        return predict_class.numpy()[0]
