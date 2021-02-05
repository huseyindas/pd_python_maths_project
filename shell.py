import basic

while True:
		text = input('Hesaplamamızı istediğiniz işlemi giriniz : ')
		result, error = basic.run('<stdin>','deneme : ', text)

		if error: print(error.as_string())
		else: print(result)
