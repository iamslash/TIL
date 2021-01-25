# Materials

* [alpine qemu](https://wiki.alpinelinux.org/wiki/Qemu)

# Install

## Install on windows

* [Download QEMU @ qemu](https://www.qemu.org/download/#windows)

Launch linux-0.0.1-remake. [linux kernel @ TIL](/linuxkernel/README.md)


# Basic Usages

```bash
# create hard disc image
qemu-img create -f qcow2 hda.img 16G
# install linux
qemu-system-i386 -hda hda.img -cdrom debian-5010-i386-netinst.iso -boot d -net nic -net user -m 512 -rtc base=localtime
# run linux
qemu hda.img -boot c -net nic -net user -m 512 -rtc base=localtime