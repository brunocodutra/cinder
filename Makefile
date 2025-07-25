CRATE_VERSION := 0.2.0
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
	@printf "position fen r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r3qbrk/6p1/2b2pPp/p3pP1Q/PpPpP2P/3P1B2/2PB3K/R5R1 w - - 16 42\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 6k1/1R3p2/6p1/2Bp3p/3P2q1/P7/1P2rQ1K/5R2 b - - 4 44\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/8/1p2k1p1/3p3p/1p1P1P1P/1P2PK2/8/8 w - - 3 54\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 7r/2p3k1/1p1p1qp1/1P1Bp3/p1P2r1P/P7/4R3/Q4RK1 w - - 0 36\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 3r3k/2r4p/1p1b3q/p4P2/P2Pp3/1B2P3/3BQ1RP/6K1 w - - 3 87\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 2r4r/1p4k1/1Pnp4/3Qb1pq/8/4BpPp/5P2/2RR1BK1 w - - 0 42\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 4q1bk/6b1/7p/p1p4p/PNPpP2P/KN4P1/3Q4/4R3 b - - 0 37\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 2q3r1/1r2pk2/pp3pp1/2pP3p/P1Pb1BbP/1P4Q1/R3NPP1/4R1K1 w - - 2 34\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 1r2r2k/1b4q1/pp5p/2pPp1p1/P3Pn2/1P1B1Q1P/2R3P1/4BR1K b - - 1 37\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r3kbbr/pp1n1p1P/3ppnp1/q5N1/1P1pP3/P1N1B3/2P1QP2/R3KB1R b KQkq b3 0 17\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/6pk/2b1Rp2/3r4/1R1B2PP/P5K1/8/2r5 b - - 16 42\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 1r4k1/4ppb1/2n1b1qp/pB4p1/1n1BP1P1/7P/2PNQPK1/3RN3 w - - 8 29\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/p2B4/PkP5/4p1pK/4Pb1p/5P2/8/8 w - - 29 68\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 3r4/ppq1ppkp/4bnp1/2pN4/2P1P3/1P4P1/PQ3PBP/R4K2 b - - 2 20\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 5rr1/4n2k/4q2P/P1P2n2/3B1p2/4pP2/2N1P3/1RR1K2Q w - - 1 49\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 1r5k/2pq2p1/3p3p/p1pP4/4QP2/PP1R3P/6PK/8 w - - 1 51\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen q5k1/5ppp/1r3bn1/1B6/P1N2P2/BQ2P1P1/5K1P/8 b - - 2 34\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r1b2k1r/5n2/p4q2/1ppn1Pp1/3pp1p1/NP2P3/P1PPBK2/1RQN2R1 w - - 0 22\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r1bqk2r/pppp1ppp/5n2/4b3/4P3/P1N5/1PP2PPP/R1BQKB1R w KQkq - 0 5\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r1bqr1k1/pp1p1ppp/2p5/8/3N1Q2/P2BB3/1PP2PPP/R3K2n b Q - 1 12\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r1bq2k1/p4r1p/1pp2pp1/3p4/1P1B3Q/P2B1N2/2P3PP/4R1K1 b - - 2 19\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r4qk1/6r1/1p4p1/2ppBbN1/1p5Q/P7/2P3PP/5RK1 w - - 2 25\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r7/6k1/1p6/2pp1p2/7Q/8/p1P2K1P/8 w - - 0 32\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r3k2r/ppp1pp1p/2nqb1pn/3p4/4P3/2PP4/PP1NBPPP/R2QK1NR w KQkq - 1 5\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 3r1rk1/1pp1pn1p/p1n1q1p1/3p4/Q3P3/2P5/PP1NBPPP/4RRK1 w - - 0 12\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 5rk1/1pp1pn1p/p3Brp1/8/1n6/5N2/PP3PPP/2R2RK1 w - - 2 20\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/1p2pk1p/p1p1r1p1/3n4/8/5R2/PP3PPP/4R1K1 b - - 3 27\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/4pk2/1p1r2p1/p1p4p/Pn5P/3R4/1P3PP1/4RK2 w - - 1 33\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/5k2/1pnrp1p1/p1p4p/P6P/4R1PK/1P3P2/4R3 b - - 1 38\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/8/1p1kp1p1/p1pr1n1p/P6P/1R4P1/1P3PK1/1R6 b - - 15 45\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/8/1p1k2p1/p1prp2p/P2n3P/6P1/1P1R1PK1/4R3 b - - 5 49\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/8/1p4p1/p1p2k1p/P2npP1P/4K1P1/1P6/3R4 w - - 6 54\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/8/1p4p1/p1p2k1p/P2n1P1P/4K1P1/1P6/6R1 b - - 6 59\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/5k2/1p4p1/p1pK3p/P2n1P1P/6P1/1P6/4R3 b - - 14 63\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 8/1R6/1p1K1kp1/p6p/P1p2P1P/6P1/1Pn5/8 w - - 0 67\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 1rb1rn1k/p3q1bp/2p3p1/2p1p3/2P1P2N/PP1RQNP1/1B3P2/4R1K1 b - - 4 23\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 4rrk1/pp1n1pp1/q5p1/P1pP4/2n3P1/7P/1P3PB1/R1BQ1RK1 w - - 3 22\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r2qr1k1/pb1nbppp/1pn1p3/2ppP3/3P4/2PB1NN1/PP3PPP/R1BQR1K1 w - - 4 12\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 2r2k2/8/4P1R1/1p6/8/P4K1N/7b/2B5 b - - 0 55\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 6k1/5pp1/8/2bKP2P/2P5/p4PNb/B7/8 b - - 1 44\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 2rqr1k1/1p3p1p/p2p2p1/P1nPb3/2B1P3/5P2/1PQ2NPP/R1R4K w - - 3 25\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen r1b2rk1/p1q1ppbp/6p1/2Q5/8/4BP2/PPP3PP/2KR1B1R b - - 2 14\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 6r1/5k2/p1b1r2p/1pB1p1p1/1Pp3PP/2P1R1K1/2P2P2/3R4 w - - 1 36\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 2rr2k1/1p4bp/p1q1p1p1/4Pp1n/2PB4/1PN3P1/P3Q2P/2RR2K1 w - f6 0 20\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 3br1k1/p1pn3p/1p3n2/5pNq/2P1p3/1PN3PP/P2Q1PB1/4R1K1 w - - 0 23\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
	@printf "position fen 2r2b2/5p2/5k2/p1r1pP2/P2pB3/1P3P2/K1P3R1/7R w - - 23 93\ngo depth 16" | $(TARGET_DIR)/$2/$1/release/cinder$(POSTFIX)
endef
