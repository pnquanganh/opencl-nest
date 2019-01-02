#ifndef __GSL_TYPES_H__
#define __GSL_TYPES_H__

#ifndef GSL_VAR

#ifdef WIN32
#  ifdef GSL_DLL
#    ifdef DLL_EXPORT
#      define GSL_VAR extern __declspec(dllexport)
#    else
#      define GSL_VAR extern __declspec(dllimport)
#    endif
#  else
#    define GSL_VAR extern
#  endif
#else
#  define GSL_VAR extern
#endif

#endif

#endif /* __GSL_TYPES_H__ */
