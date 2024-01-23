import pdfplumber
import chardet
pdf = pdfplumber.open("UNECE 贸易便利化术语 TRADE Facilitation Terms v3 cn en ru.pdf")
chshead=[]
chs=[]
chsheadtemp=""
chstemp=""
switch=0
# for page in pdf.pages[29:273]:
for page in pdf.pages[29:273]:
    for y in page.chars:
        # print(y["text"]+":"+y["fontname"])
        if (y["fontname"]=="SRKVRY+NotoSansSC-Bold"): 
            if (switch==1):
                # print(chstemp)
                chs.append(chstemp)
                switch=0
                chstemp=''     
            elif (switch==2):
                chs.append('') 
                switch=0  
            chsheadtemp+=y["text"]
            # print('head:'+y["text"]+":"+y["fontname"])
        elif (y["fontname"]=="XWAZUO+NotoSansSC-Light" or y['fontname']=='GMDXAW+NotoSansSC-Thin'):
            if (switch==0):
                # print(chsheadtemp+":")
                chshead.append(chsheadtemp) 
                switch=1
                chsheadtemp=''
            elif (switch==2):
                switch=1
            chstemp+=y["text"]
            # print('list:'+ y["text"]+":"+y["fontname"])
        else:
            if (switch==0):
                chshead.append(chsheadtemp) 
                switch=2
                chsheadtemp=''
chs.append(chstemp)
del(chshead[0])
del(chs[0])
# print(len(chshead))
# print(len(chs))
# for i in range(100):
#     print(chshead[i]+":"+chs[i]) 

# print(chshead[823]+":"+chs[823])
import pandas as pd
table=[['question','answer']]
for i in range(len(chshead)):
    temp = [[chshead[i],chs[i]]]
    table.extend(temp)
df = pd.DataFrame(table[1:],columns=table[0])
df.to_excel('贸易便利化术语.xlsx',index=False)