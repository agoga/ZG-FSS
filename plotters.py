import config as cfg

setname='E6W16MINZ3MM.txt'

with open(csvfilename,'a',newline='', encoding='utf-8') as fd:
        csv_writer=writer(fd)
        csv_writer.writerow(data)


