CRATE_VERSION := 0.5.2
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

native-rustflags := "-Ctarget-cpu=native"
sse4-rustflags := "-Ctarget-cpu=x86-64-v2"
avx2-rustflags := "-Ctarget-cpu=x86-64-v3","-Ztune-cpu=znver3"
avx512-rustflags := "-Ctarget-cpu=x86-64-v4","-Ztune-cpu=znver4","-Ctarget-feature=+gfni,+avx512ifma,+avx512bitalg,+avx512vbmi,+avx512vbmi2,+avx512vnni,+avx512vpopcntdq"

default: native

spsa:
	$(call build,spsa,$(shell rustc --print host-tuple),$(native-rustflags),--features spsa)

native:
	$(call build,native,$(shell rustc --print host-tuple),$(native-rustflags),)

linux-aarch64-neon:
	$(call build,neon,aarch64-unknown-linux-musl,,)

linux-x86-64-sse4:
	$(call build,sse4,x86_64-unknown-linux-gnu,$(sse4-rustflags),)

linux-x86-64-avx2:
	$(call build,avx2,x86_64-unknown-linux-gnu,$(avx2-rustflags),)

linux-x86-64-avx512:
	$(call build,avx512,x86_64-unknown-linux-gnu,$(avx512-rustflags),)

windows-aarch64-neon:
	$(call build,neon,aarch64-pc-windows-msvc,,)

windows-x86-64-sse4:
	$(call build,sse4,x86_64-pc-windows-msvc,$(sse4-rustflags),)

windows-x86-64-avx2:
	$(call build,avx2,x86_64-pc-windows-msvc,$(avx2-rustflags),)

windows-x86-64-avx512:
	$(call build,avx512,x86_64-pc-windows-msvc,$(avx512-rustflags),)

mac-aarch64-neon:
	$(call build,neon,aarch64-apple-darwin,,)

.PHONY: default spsa native
.PHONY: linux-aarch64-neon
.PHONY: linux-x86-64-sse4 linux-x86-64-avx2 linux-x86-64-avx512
.PHONY: windows-aarch64-neon
.PHONY: windows-x86-64-sse4 windows-x86-64-avx2 windows-x86-64-avx512
.PHONY: mac-aarch64-neon

define build
	@echo "Building target $1"
	rustup target add $2
	cargo build --profile=dist --bin=cinder \
		--config='target.$2.rustflags=["-Zlocation-detail=none",$3]' \
		--target-dir=$(TARGET_DIR)/$1/ --target=$2 $4

	@mkdir -p $(BIN_DIR)
	@cp $(TARGET_DIR)/$1/$2/dist/cinder$(POSTFIX) $(BIN_DIR)/cinder-v$(CRATE_VERSION)-$(PLATFORM)-$1$(POSTFIX)
endef
