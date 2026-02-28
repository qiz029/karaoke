.PHONY: package
package:
	@pyinstaller --onefile --clean --name "ktv_bff" main.py

# test
.PHONY: dev
dev:
	@python main.py