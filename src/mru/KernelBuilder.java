package mru;

public class KernelBuilder {
 static String create(int mapDimX, int mapDimY, int memSize) {
 	final String kernelSource =
 		"const int dx[] = { 0, -1, 0, 1 }; " +
 		"const int dy[] = { -1, 0, 1, 0 }; " +
 		// nop
 		// unconditional up, left, down, right, 
 		// conditional up, left, down, right, 
 		// incmem, decmem, incptr, decptr
 		"const int cond[]   = {0,  1, 1, 1, 1,  2, 2, 2, 2,  0, 0, 0, 0}; " +
 		"const int newdir[] = {0,  0, 1, 2, 3,  0, 1, 2, 3,  0, 0, 0, 0}; " +
 		"const int incmem[] = {0,  0, 0, 0, 0,  0, 0, 0, 0,  1,-1, 0, 0}; " +
 		"const int incptr[] = {0,  0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 1,-1}; " +
 		"\n" +
 		"kernel void " + 
 		"adv(global int *map, " +
 		"    global int *x, " +
 		"    global int *y, " +
 		"    global int *dir, " +
 		"    global int *mem, " +
 		"    global int *ptr) { " +
 		" unsigned int xid = get_global_id(0); " +
 		"\n" +
 		" int pixel = map[x[xid] * " + mapDimX + " + y[xid]]; " +
 		"\n" +
 		" int ndircmpres = select(dir[xid], newdir[pixel], mem[ptr[xid]]); " +
 		" int pndir = select(newdir[pixel], ndircmpres, cond[pixel] >> 1); " +
 		" dir[xid] = select(dir[xid], pndir, cond[pixel]); " +
 		" mem[ptr[xid]] += incmem[pixel]; " +
 		" mem[ptr[xid]] = max(0, mem[ptr[xid]]); " +
 		" ptr[xid] += incptr[pixel]; " +
 		" ptr[xid] = (ptr[xid] + " + memSize + ") % " + memSize + "; " +
 		"\n" +
 		" x[xid] += dx[dir[xid]] + " + mapDimX + "; " +
 		" y[xid] += dy[dir[xid]] + " + mapDimY + "; " +
 		" x[xid] %= " + mapDimX + "; " +
 		" y[xid] %= " + mapDimY + "; " +
 		"}";
 	return kernelSource;
 }
}
