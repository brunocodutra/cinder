CRATE_VERSION := 0.4.1
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

default: native

spsa:
	$(call build,native,$(shell rustc --print host-tuple),native,--features spsa)

native:
	$(call build,native,$(shell rustc --print host-tuple),native,)

linux-x86-64-sse4:
	$(call build,sse4,x86_64-unknown-linux-gnu,x86-64-v2,)

linux-x86-64-avx2:
	$(call build,avx2,x86_64-unknown-linux-gnu,x86-64-v3,)

linux-x86-64-avx512:
	$(call build,avx512,x86_64-unknown-linux-gnu,x86-64-v4,)

linux-x86-64-vnni512:
	$(call build,vnni512,x86_64-unknown-linux-gnu,znver5,) # placeholder for x86-64-v5

windows-x86-64-sse4:
	$(call build,sse4,x86_64-pc-windows-msvc,x86-64-v2,)

windows-x86-64-avx2:
	$(call build,avx2,x86_64-pc-windows-msvc,x86-64-v3,)

windows-x86-64-avx512:
	$(call build,avx512,x86_64-pc-windows-msvc,x86-64-v4,)

windows-x86-64-vnni512:
	$(call build,vnni512,x86_64-pc-windows-msvc,znver5,) # placeholder for x86-64-v5

mac-aarch64-neon:
	$(call build,neon,aarch64-apple-darwin,apple-m1,)

mac-aarch64-sme:
	$(call build,sme,aarch64-apple-darwin,apple-m4,)

.PHONY: default spsa native
.PHONY: linux-x86-64-sse4 linux-x86-64-avx2 linux-x86-64-avx512 linux-x86-64-vnni512
.PHONY: windows-x86-64-sse4 windows-x86-64-avx2 windows-x86-64-avx512 windows-x86-64-vnni512
.PHONY: mac-aarch64-neon mac-aarch64-sme

define build
	@echo "Building target $1"
	cargo build --release --bin=cinder -Zunstable-options -Zprofile-rustflags \
		--config=profile.release.rustflags='["-Ctarget-cpu=$3", "-Cpanic=abort", "-Cstrip=symbols", "-Zlocation-detail=none"]' \
		--target-dir=$(TARGET_DIR)/$1/ --target=$2 $4

	@mkdir -p $(BIN_DIR)
	@cp $(TARGET_DIR)/$1/$2/release/cinder$(POSTFIX) $(BIN_DIR)/cinder-v$(CRATE_VERSION)-$(PLATFORM)-$1$(POSTFIX)
endef
