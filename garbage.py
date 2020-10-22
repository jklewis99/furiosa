errors = []

try:
    hi = []
    print(hi.nope)
except Exception as e:
    errors.append(str(e))

f = open('errors/trailer-database-errors.txt', 'w+')
f.writelines(errors)