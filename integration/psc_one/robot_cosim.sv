`timescale 1ns/1ps
// TEST ONLY: bypass the existing Chip/SoC pin routing defects. This is NOT
// validation of board pins. CPU, MMIO and the I2C slave itself are unchanged.
module robot_cosim (
    input wire clock, rst, scl_master, master_sda_low,
    output wire sda_bus
);
    PSC_ONE_Chip_sim system (
        .clock(clock), .rst(rst), .uart_rx(1'b1), .uart_tx(),
        .PSCONE_SCL(), .PSCONE_SDA(),
        .PSCONE_SW1(1'b0), .PSCONE_SW2(1'b0), .led_out()
    );
    assign sda_bus = ~(master_sda_low | system.u_chip.u_soc.u_i2c.sda_drive_low);
    always @(*) begin
        if (scl_master) force system.u_chip.u_soc.u_i2c.scl_i = 1'b1;
        else force system.u_chip.u_soc.u_i2c.scl_i = 1'b0;
        if (sda_bus) force system.u_chip.u_soc.u_i2c.sda_i = 1'b1;
        else force system.u_chip.u_soc.u_i2c.sda_i = 1'b0;
    end
endmodule
