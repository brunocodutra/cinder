CRATE_VERSION := 0.3.0
TARGET_DIR := $(CURDIR)/target
BIN_DIR := $(TARGET_DIR)/bin

ifeq ($(OS),Windows_NT)
	PLATFORM := windows
	POSTFIX := .exe
else
	UNAME := $(shell uname -s)
	ifeq ($(UNAME),Linux)
		PLATFORM := linux
		POSTFIX :=
	else ifeq ($(UNAME),Darwin)
		PLATFORM := mac
		POSTFIX :=
	endif
endif

default: cinder-native

spsa:
	$(call build,$(shell rustc --print host-tuple),native,--features spsa)

cinder-native:
	$(call build,$(shell rustc --print host-tuple),native,)

cinder-linux-x86-64:
	$(call build,x86_64-unknown-linux-gnu,x86-64,)

cinder-linux-x86-64-v2:
	$(call build,x86_64-unknown-linux-gnu,x86-64-v2,)

cinder-linux-x86-64-v3:
	$(call build,x86_64-unknown-linux-gnu,x86-64-v3,)

cinder-linux-x86-64-v4:
	$(call build,x86_64-unknown-linux-gnu,x86-64-v4,)

cinder-windows-x86-64:
	$(call build,x86_64-pc-windows-msvc,x86-64,)

cinder-windows-x86-64-v2:
	$(call build,x86_64-pc-windows-msvc,x86-64-v2,)

cinder-windows-x86-64-v3:
	$(call build,x86_64-pc-windows-msvc,x86-64-v3,)

cinder-windows-x86-64-v4:
	$(call build,x86_64-pc-windows-msvc,x86-64-v4,)

cinder-mac-apple-m1:
	$(call build,aarch64-apple-darwin,apple-m1,)

cinder-mac-apple-m2:
	$(call build,aarch64-apple-darwin,apple-m2,)

cinder-mac-apple-m3:
	$(call build,aarch64-apple-darwin,apple-m3,)

cinder-mac-apple-m4:
	$(call build,aarch64-apple-darwin,apple-m4,)

release:
ifeq ($(PLATFORM),linux)
	$(MAKE) cinder-linux-x86-64 cinder-linux-x86-64-v2 cinder-linux-x86-64-v3 cinder-linux-x86-64-v4
else ifeq ($(PLATFORM),windows)
	$(MAKE) cinder-windows-x86-64 cinder-windows-x86-64-v2 cinder-windows-x86-64-v3 cinder-windows-x86-64-v4
else ifeq ($(PLATFORM),mac)
	$(MAKE) cinder-mac-apple-m1 cinder-mac-apple-m2 cinder-mac-apple-m3 cinder-mac-apple-m4
else
	@echo "Unsupported platform $(PLATFORM)"
endif

.PHONY: default release spsa cinder-native
.PHONY: cinder-linux-x86-64 cinder-linux-x86-64-v2 cinder-linux-x86-64-v3 cinder-linux-x86-64-v4
.PHONY: cinder-windows-x86-64 cinder-windows-x86-64-v2 cinder-windows-x86-64-v3 cinder-windows-x86-64-v4
.PHONY: cinder-mac-x86-64 cinder-mac-x86-64-v2 cinder-mac-x86-64-v3 cinder-mac-x86-64-v4
.PHONY: cinder-mac-apple-m1 cinder-mac-apple-m2 cinder-mac-apple-m3 cinder-mac-apple-m4

define build
	@echo "Building target $1 for CPU architecture $2"
	$(call instrument,$1,$2,$3);
	$(call profile,$1,$2);
	$(call optimize,$1,$2,$3);
	$(call release,$1,$2);
endef

define release
	@mkdir -p $(BIN_DIR)
	@cp $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX) $(BIN_DIR)/cinder-v$(CRATE_VERSION)-$(PLATFORM)-$2$(POSTFIX)
endef

define instrument
	cargo pgo instrument build -- \
		--bin=cinder \
		-Zunstable-options \
		--config=profile.release.rustflags='["-Ctarget-cpu=$2"]' \
		-Zbuild-std=core,alloc,std,panic_abort \
		-Zbuild-std-features=panic_immediate_abort \
		--target-dir=$(TARGET_DIR)/$2/ \
		--target=$1 $3
endef

define optimize
	cargo pgo optimize build -- \
		--bin=cinder \
		-Zunstable-options \
		--config=profile.release.rustflags='["-Ctarget-cpu=$2"]' \
		-Zbuild-std=core,alloc,std,panic_abort \
		-Zbuild-std-features=panic_immediate_abort \
		--target-dir=$(TARGET_DIR)/$2/ \
		--target=$1 $3
endef

define profile
	echo "=== BINARY EXISTENCE AND PERMISSIONS ==="
	ls -la $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	file $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)

	echo -e "\n=== DEPENDENCY CHECK ==="
	ldd $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX) 2>/dev/null || objdump -p $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX) | grep "DLL Name" || echo "Could not check dependencies"

	echo -e "\n=== DIRECT EXECUTION TEST ==="
	timeout 5s $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX) || echo "Exit code: $?"

	echo -e "\n=== QUIT COMMAND TEST ==="
	printf "quit\n" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	echo "Exit code after quit: $?"

	echo -e "\n=== SIMPLE POSITION TEST ==="
	printf "position startpos\ngo depth 1\nquit\n" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	echo "Exit code after simple position: $?"

	echo -e "\n=== ORIGINAL FAILING COMMAND TEST ==="
	printf "position fen r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14\ngo depth 16\n" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	echo "Exit code after original command: $?"

	echo -e "\n=== PRINTF TEST ==="
	printf "position fen r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14\ngo depth 16"
	echo -e "\n^ That was the printf output"

	echo -e "\n=== ENVIRONMENT INFO ==="
	echo "Shell: $(SHELL)"
	echo "PATH: $(PATH)"
	echo "Working directory: $(pwd)"
	echo "Make version: $(make --version | head -1)"

	echo -e "\n=== STDERR CAPTURE TEST ==="
	printf "position fen r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14\ngo depth 16\n" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX) 2>&1
	echo "Exit code with stderr captured: $?"
endef
