/* Integration-only boot fixture: pinned PSC-ONE omits psc_mbios.S.
 * Enter the unmodified OS boot function in S mode. No SBI, warm restart,
 * interrupts or user processes are supported by this dedicated service image.
 */
.section .text
.globl _start
_start:
    csrw satp, zero
    csrw mie, zero
    la t0, trap
    csrw mtvec, t0
    csrw stvec, t0
    li t0, 0x800
    csrw mstatus, t0
    li t0, 0x00200000
    csrw mepc, t0
    mret
trap:
    j trap
