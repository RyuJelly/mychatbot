class FindAnswer:
    def __init__(self, db):
        self.db = db

    # 검색 쿼리 생성
    def _make_query(self, intent_name):
        sql = "select from ANSWER_COUNT"
        if intent_name != None:
            sql = sql + f" where intent='{intent_name}' "

        return sql

        # 의도명으로 답변 검색
        sql = self._make_query(intent_name)
        answer = self.db.select_one(sql)

        return (answer['answer'])

    # # NER 태그를 실제 입력된 단어로 변환
    # def tag_to_word(self, ner_predicts, answer):
    #     for word, tag in ner_predicts:
    #
    #         # 변환해야하는 태그가 있는 경우 추가
    #         if tag == 'B_FOOD' or tag == 'B_DT' or tag == 'B_TI':
    #             answer = answer.replace(tag, word)
    #
    #     answer = answer.replace('{', '')
    #     answer = answer.replace('}', '')
    #     return answer
