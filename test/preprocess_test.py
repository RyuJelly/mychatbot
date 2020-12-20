from utils.Preprocess import Preprocess

sent = "냉장고에 있는 재료로 레시피 추천해줄래?"

p = Preprocess(userdic='../utils/user_dic.tsv')

pos = p.pos(sent)

ret = p.get_keywords(pos, without_tag=False)
print(ret)

ret = p.get_keywords(pos, without_tag=True)
print(ret)