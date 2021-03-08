# Abstract

UNIX V6 의 source 를 분석한다.

# Materials

* [Linux Kernel Boot Process by 조만석 pdf](https://www.sosconhistory.net/2015/download/day28/ST3/S_28_1000_%EC%A1%B0%EB%A7%8C%EC%84%9D.pdf)
  * [UNIX V6로 배우는 커널의 원리와 구조 @ yes24](http://www.yes24.com/Product/Goods/12982537)
  * [The Unix Tree](https://minnie.tuhs.org/cgi-bin/utree.pl)
  * [Unix V6 Manuals](http://man.cat-v.org/unix-6th/)
* [Xv6, a simple Unix-like teaching operating system @ mit](https://pdos.csail.mit.edu/6.828/2012/xv6.html)
  * unix v6 를 x86 실행환경으로 다시 개발한 것
  * `git clone git://github.com/mit-pdos/xv6-public.git`
  * [textbook/commentary pdf](https://pdos.csail.mit.edu/6.828/2012/xv6/book-rev7.pdf)
* [Commentary on the Sixth Edition UNIX Operating System](http://www.lemis.com/grog/Documentation/Lions/)
  * [src @ minnie](https://minnie.tuhs.org/cgi-bin/utree.pl?file=V6/usr)
  * [src @ github](https://github.com/memnoth/unix-v6)

# Prerequisites

> * [src @ github](https://github.com/memnoth/unix-v6)
> * [UNIX Assembler Reference Manual](http://www.tom-yam.or.jp/2238/ref/as.pdf)
> * [Unix V6 Manuals](http://man.cat-v.org/unix-6th/)
> * [Commentary on the Sixth Edition UNIX Operating System](http://www.lemis.com/grog/Documentation/Lions/)
> * [The PDP-11 Assembly Language](https://programmer209.wordpress.com/2011/08/03/the-pdp-11-assembly-language/)

# PSW (Process Status Word)

# Global Registers

| Name | Description |
|--|--|
| r0 | |
| r1 | |
| r2 | |
| r3 | |
| r4 | |
| r5 | |
| r6 | |
| r7 | |

# Process Memory Structure

Text Segment

Data Segment 

* PPDA (Per Process Data Area)
* Data Area
* Stack Area

# proc struct

# user struct

# Memory Management Status Regiters

SR0

SR2

# Virtual Address to Physical Address Translation

# Pre K&R

# Assembly Syntax

> * [The PDP-11 Addressing Modes](https://programmer209.wordpress.com/2011/08/03/the-pdp-11-assembly-language/)

| Syntax | Mode | Action | Machine Code | Extra Word |
|--|--|--|--|--|
| Rn | Register | Data = Rn | 0n | - |
| (Rn)+ | Autoincrement | Data = (Rn), Rn++ | 2n | - |
| -(Rn) | Autodecrement | | | |
| X(Rn) | Index | | | |
| @Rn or (Rn) | Register Deferred | | | |
| @(Rn)+ | Autoincrement Deferred | | | |
| @-(Rn) | Autodecrement Deferred | | | |
| @X(Rn) | Index Deferred | | | |
| #n | Immediate | | | |
| @#A | Immediate Deferred (Absolute) | | | |
| A or X(PC) | Relative | | | |
| @A or @X(PC) | Relative Deferred | | | |

# fork systemcall

> `source/s4/fork.s`

```s
/ C library -- fork

/ pid = fork();
/
/ pid == 0 in child process; pid == -1 means error return
/ in child, parents id is in par_uid if needed

.globl	_fork, cerror, _par_uid

_fork:
	mov	r5,-(sp)
	mov	sp,r5
	sys	fork
		br 1f
	bec	2f
	jmp	cerror
1:
	mov	r0,_par_uid
	clr	r0
2:
	mov	(sp)+,r5
	rts	pc
.bss
_par_uid: .=.+2
```

> `sys/ken/sys1.c`

```c
fork()
{
	register struct proc *p1, *p2;

	p1 = u.u_procp;
	for(p2 = &proc[0]; p2 < &proc[NPROC]; p2++)
		if(p2->p_stat == NULL)
			goto found;
	u.u_error = EAGAIN;
	goto out;

found:
	if(newproc()) {
		u.u_ar0[R0] = p1->p_pid;
		u.u_cstime[0] = 0;
		u.u_cstime[1] = 0;
		u.u_stime = 0;
		u.u_cutime[0] = 0;
		u.u_cutime[1] = 0;
		u.u_utime = 0;
		return;
	}
	u.u_ar0[R0] = p2->p_pid;

out:
	u.u_ar0[R7] =+ 2;
}
```

> `sys/ken/slp.c`

```c
/*
 * Create a new process-- the internal version of
 * sys fork.
 * It returns 1 in the new process.
 * How this happens is rather hard to understand.
 * The essential fact is that the new process is created
 * in such a way that appears to have started executing
 * in the same call to newproc as the parent;
 * but in fact the code that runs is that of swtch.
 * The subtle implication of the returned value of swtch
 * (see above) is that this is the value that newproc's
 * caller in the new process sees.
 */
newproc()
{
	int a1, a2;
	struct proc *p, *up;
	register struct proc *rpp;
	register *rip, n;

	p = NULL;
	/*
	 * First, just locate a slot for a process
	 * and copy the useful info from this process into it.
	 * The panic "cannot happen" because fork has already
	 * checked for the existence of a slot.
	 */
retry:
	mpid++;
	if(mpid < 0) {
		mpid = 0;
		goto retry;
	}
	for(rpp = &proc[0]; rpp < &proc[NPROC]; rpp++) {
		if(rpp->p_stat == NULL && p==NULL)
			p = rpp;
		if (rpp->p_pid==mpid)
			goto retry;
	}
	if ((rpp = p)==NULL)
		panic("no procs");

	/*
	 * make proc entry for new proc
	 */

	rip = u.u_procp;
	up = rip;
	rpp->p_stat = SRUN;
	rpp->p_flag = SLOAD;
	rpp->p_uid = rip->p_uid;
	rpp->p_ttyp = rip->p_ttyp;
	rpp->p_nice = rip->p_nice;
	rpp->p_textp = rip->p_textp;
	rpp->p_pid = mpid;
	rpp->p_ppid = rip->p_pid;
	rpp->p_time = 0;

	/*
	 * make duplicate entries
	 * where needed
	 */

	for(rip = &u.u_ofile[0]; rip < &u.u_ofile[NOFILE];)
		if((rpp = *rip++) != NULL)
			rpp->f_count++;
	if((rpp=up->p_textp) != NULL) {
		rpp->x_count++;
		rpp->x_ccount++;
	}
	u.u_cdir->i_count++;
	/*
	 * Partially simulate the environment
	 * of the new process so that when it is actually
	 * created (by copying) it will look right.
	 */
	savu(u.u_rsav);
	rpp = p;
	u.u_procp = rpp;
	rip = up;
	n = rip->p_size;
	a1 = rip->p_addr;
	rpp->p_size = n;
	a2 = malloc(coremap, n);
	/*
	 * If there is not enough core for the
	 * new process, swap out the current process to generate the
	 * copy.
	 */
	if(a2 == NULL) {
		rip->p_stat = SIDL;
		rpp->p_addr = a1;
		savu(u.u_ssav);
		xswap(rpp, 0, 0);
		rpp->p_flag =| SSWAP;
		rip->p_stat = SRUN;
	} else {
	/*
	 * There is core, so just copy.
	 */
		rpp->p_addr = a2;
		while(n--)
			copyseg(a1++, a2++);
	}
	u.u_procp = rip;
	return(0);
}
```


