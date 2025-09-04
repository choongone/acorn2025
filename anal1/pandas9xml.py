# XML 문서 처리
from bs4 import BeautifulSoup

with open('anal1/my.xml', mode='r', encoding='utf-8') as f:    # utf8이라고 써도 상관X
    xmlfile = f.read()
    # print(xmlfile)
soup = BeautifulSoup(xmlfile, 'lxml')
# print(soup.prettify())  # 이쁘게 보여달라는 뜻 
itemTag = soup.find_all('item')
# print(itemTag)
print()
nameTag = soup.find_all('name')
print(nameTag[0]['id'])

print('--------------')
for i in itemTag:
    nameTag = soup.find_all('name')
    for j in nameTag:
        print('id' + j['id'] + ', name:' + j.string)
        tel = i.find('tel')
        print('tel:' + tel.string)
    for j in i.findAll('exam'):
        print('kor:' + j['kor'] + ', end:' + j['eng'])
    print()
    

    