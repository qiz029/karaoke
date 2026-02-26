.PHONY: package
package:
	@pyinstaller --onefile --clean --name "ktv_bff" main.py

.PHONY: dev
dev:
	@python main.py