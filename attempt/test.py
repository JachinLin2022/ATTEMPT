import datasets
mrqa = datasets.load_dataset('tau/mrqa','newsqa',split='train')
print(mrqa[0]['answer'])