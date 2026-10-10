# Reuse the pinned PSC-ONE source list; all outputs live in PSC_Robot/build.
.DEFAULT_GOAL := cosim
ROOT := $(abspath ../../../..)
INTEGRATION := $(ROOT)/integration/psc_one
OUT ?= $(ROOT)/build/psc-one
include Makefile.pscos

.PHONY: cosim
cosim:
	$(MAKE) -C ../../software/os Build=$(OUT)/os SIM_USER=minimal SRCS_MBIOS=$(INTEGRATION)/boot.S KCPPFLAGS="-DBSS_CLEAR_OFF=1 -DPSC_ROBOT_SIM=1" sim_mem
	mkdir -p $(OUT)/mem $(OUT)/wave
	cp $(OUT)/os/bootrom.mem $(OUT)/os/kernel.mem $(OUT)/os/user.mem $(OUT)/mem/
	# Declared but unused by direct OS boot; avoid a missing-file warning.
	printf '00000000\n' > $(OUT)/mem/bootloader.mem
	cd $(OUT) && PYTHONPATH="$(INTEGRATION):$(abspath .):$$PYTHONPATH" \
	COCOTB_TEST_MODULES=cosim TOPLEVEL=robot_cosim TOPLEVEL_LANG=verilog SIM=verilator \
	VERILOG_SOURCES="$(abspath $(INPUT_FILE_PSCONE)) $(INTEGRATION)/robot_cosim.sv" \
	COMPILE_ARGS="$(TOP_COMPILE_ARGS)" \
	EXTRA_ARGS="$(EXTRA_ARGS) $(VDEF)" \
	$(MAKE) -f $(COCOTB_MAKEFILE) SIM_BUILD=$(OUT)/rtl COCOTB_RESULTS_FILE=$(OUT)/results.xml

# Existing standalone regression, isolated from normal PSC-ONE build outputs.
.PHONY: i2c-tests
i2c-tests:
	mkdir -p $(OUT)/i2c
	cd $(OUT)/i2c && PYTHONPATH="$(abspath .):$$PYTHONPATH" \
	COCOTB_TEST_MODULES=cocotb_tb.mmio.psc_rv32_i2c \
	TOPLEVEL=sim_PSC_RV32_I2C TOPLEVEL_LANG=verilog SIM=icarus \
	VERILOG_SOURCES="$(abspath $(I2C_DIR)/sim_PSC_RV32_I2C.sv $(I2C_DIR)/PSC_RV32_I2C.sv)" \
	$(MAKE) -f $(COCOTB_MAKEFILE) SIM_BUILD=$(OUT)/i2c/rtl
