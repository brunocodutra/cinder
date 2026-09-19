CRATE_VERSION := 0.6.1
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

profile ?= dist
native-rustflags := "-Ctarget-cpu=native"
sse4-rustflags := "-Ctarget-cpu=x86-64-v2"
avx2-rustflags := "-Ctarget-cpu=x86-64-v3","-Ztune-cpu=znver3"
avx512-rustflags := "-Ctarget-cpu=x86-64-v4","-Ztune-cpu=znver4","-Ctarget-feature=+gfni,+avx512ifma,+avx512bitalg,+avx512vbmi,+avx512vbmi2,+avx512vnni,+avx512vpopcntdq"
neon-rustflags := "-Ctarget-feature=+dotprod"

default: native

help:
	@echo "Targets (default: native):"
	@echo "  spsa                  Engine build with SPSA tuning enabled"
	@echo "  native                Native binaries for this OS and CPU"
	@echo "  linux-aarch64-neon    Linux on aarch64 with NEON"
	@echo "  linux-x86-64-sse4     Linux on x86-64 with SSE4"
	@echo "  linux-x86-64-avx2     Linux on x86-64 with AVX2"
	@echo "  linux-x86-64-avx512   Linux on x86-64 with AVX512"
	@echo "  windows-aarch64-neon  Windows on aarch64 with NEON"
	@echo "  windows-x86-64-sse4   Windows on x86-64 with SSE4"
	@echo "  windows-x86-64-avx2   Windows on x86-64 with AVX2"
	@echo "  windows-x86-64-avx512 Windows on x86-64 with AVX512"
	@echo "  mac-aarch64-neon      macOS on aarch64 with NEON"
	@echo ""
	@echo "Variables:"
	@echo "  profile               Cargo build profile, e.g. stage (default: dist)"
	@echo "  extra-rustflags       Extra flags passed to rustc, e.g. \"-Cforce-frame-pointers=yes\" (default: <empty>)"

spsa:
	$(call build,spsa,$(shell rustc --print host-tuple),$(native-rustflags),--features spsa)

native:
	$(call build,native,$(shell rustc --print host-tuple),$(native-rustflags),)

linux-aarch64-neon:
	$(call build,neon,aarch64-unknown-linux-musl,$(neon-rustflags),)

linux-x86-64-sse4:
	$(call build,sse4,x86_64-unknown-linux-gnu,$(sse4-rustflags),)

linux-x86-64-avx2:
	$(call build,avx2,x86_64-unknown-linux-gnu,$(avx2-rustflags),)

linux-x86-64-avx512:
	$(call build,avx512,x86_64-unknown-linux-gnu,$(avx512-rustflags),)

windows-aarch64-neon:
	$(call build,neon,aarch64-pc-windows-msvc,$(neon-rustflags),)

windows-x86-64-sse4:
	$(call build,sse4,x86_64-pc-windows-msvc,$(sse4-rustflags),)

windows-x86-64-avx2:
	$(call build,avx2,x86_64-pc-windows-msvc,$(avx2-rustflags),)

windows-x86-64-avx512:
	$(call build,avx512,x86_64-pc-windows-msvc,$(avx512-rustflags),)

mac-aarch64-neon:
	$(call build,neon,aarch64-apple-darwin,$(neon-rustflags),)

.PHONY: default help spsa native
.PHONY: linux-aarch64-neon
.PHONY: linux-x86-64-sse4 linux-x86-64-avx2 linux-x86-64-avx512
.PHONY: windows-aarch64-neon
.PHONY: windows-x86-64-sse4 windows-x86-64-avx2 windows-x86-64-avx512
.PHONY: mac-aarch64-neon

define build
	@echo "Building target $1"
	rustup target add $2
	cargo build --profile=$(profile) --bin=cinder \
		--config='target.$2.rustflags=["-Zlocation-detail=none",$3,$(extra-rustflags)]' \
		--target-dir=$(TARGET_DIR)/$1/ --target=$2 $4

	@mkdir -p $(BIN_DIR)
	@cp $(TARGET_DIR)/$1/$2/$(profile)/cinder$(POSTFIX) $(BIN_DIR)/cinder-v$(CRATE_VERSION)-$(PLATFORM)-$1$(POSTFIX)
endef
