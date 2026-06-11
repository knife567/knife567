import docx

doc = docx.Document("/home/runner/work/knife567/knife567/knife567/knife567/项目资料-任.docx")
for para in doc.paragraphs:
    print(para.text)
