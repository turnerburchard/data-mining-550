from crossref.restful import Works
works = Works()

w1 = works.query(author='Caleb Eardley')

for item in w1:
    print(item)
