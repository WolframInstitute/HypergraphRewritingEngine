#pragma once
// GPUMC's <stdio.h> (printf and a few file calls), and the rest of the C library's <stdio.h>
// declarations, which libstdc++'s <cstdio> brings into namespace std. No harness calls the
// declared-only ones.
#include_next <stdio.h>

#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef long fpos_t;
#ifndef EOF
#define EOF (-1)
#endif
#ifndef BUFSIZ
#define BUFSIZ 8192
#endif
#ifndef SEEK_SET
#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2
#endif

void clearerr(FILE*);
int feof(FILE*);
int ferror(FILE*);
int fgetc(FILE*);
int fgetpos(FILE*, fpos_t*);
char* fgets(char*, int, FILE*);
int fputc(int, FILE*);
int fputs(const char*, FILE*);
size_t fread(void*, size_t, size_t, FILE*);
FILE* freopen(const char*, const char*, FILE*);
int fscanf(FILE*, const char*, ...);
int fseek(FILE*, long, int);
int fsetpos(FILE*, const fpos_t*);
long ftell(FILE*);
int getc(FILE*);
int getchar(void);
void perror(const char*);
int putc(int, FILE*);
int putchar(int);
int puts(const char*);
int remove(const char*);
int rename(const char*, const char*);
void rewind(FILE*);
int scanf(const char*, ...);
void setbuf(FILE*, char*);
int setvbuf(FILE*, char*, int, size_t);
int sprintf(char*, const char*, ...);
int snprintf(char*, size_t, const char*, ...);
int sscanf(const char*, const char*, ...);
FILE* tmpfile(void);
char* tmpnam(char*);
int ungetc(int, FILE*);
int vfprintf(FILE*, const char*, va_list);
int vprintf(const char*, va_list);
int vsprintf(char*, const char*, va_list);
int vsnprintf(char*, size_t, const char*, va_list);
int vfscanf(FILE*, const char*, va_list);
int vscanf(const char*, va_list);
int vsscanf(const char*, const char*, va_list);

#ifdef __cplusplus
}
#endif
