import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing


# 의도 분류 모델 모듈
class IntentModel:
    tf.compat.v1.enable_eager_execution()
    tf.compat.v1.logging.set_verbosity( tf.compat.v1.logging.ERROR)
    def __init__(self, model_name, proprocess):

        # 의도 클래스 별 레이블
        self.labels = {0: "달걀개수", 1: "레몬개수", 2: "자두개수", 3: "오이개수",  4: "사이다개수", 5: "당근개수", 6: "애호박개수", 7: "옥수수개수", 8: "파인애플개수",
                       9: "사과개수", 10: "양파개수", 11: "마늘개수", 12: "토마토개수", 13: "브로콜리개수", 14: "깻잎개수",
                       15: "가지개수", 16: "단호박개수", 17: "무개수", 18: "양배추개수", 19: "파프리카개수", 20: "야쿠르트개수", 21: "맥주개수", 22: "콜라개수"}

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
        # print('predict_class',predict_class)
        # print('predict_class.numpy()[0]',predict_class.numpy()[0])
        return predict_class.numpy()[0]
