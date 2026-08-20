#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

template <typename F>
inline double zero_brent(const double &a, const double &b, const double t, F &f)

//****************************************************************************80
//
//  Purpose:
//
//    ZERO seeks the root of a function F(X) in an interval [A,B].
//
//  Discussion:
//
//    The interval [A,B] must be a change of sign interval for F.
//    That is, F(A) and F(B) must be of opposite signs.  Then
//    assuming that F is continuous implies the existence of at least
//    one value C between A and B for which F(C) = 0.
//
//    The location of the zero is determined to within an accuracy
//    of 6 * MACHEPS * abs ( C ) + 2 * T.
//
//    Thanks to Thomas Secretin for pointing out a transcription error in the
//    setting of the value of P, 11 February 2013.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 February 2013
//
//  Author:
//
//    Original FORTRAN77 version by Richard Brent.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Richard Brent,
//    Algorithms for Minimization Without Derivatives,
//    Dover, 2002,
//    ISBN: 0-486-41998-3,
//    LC: QA402.5.B74.
//
//  Parameters:
//
//    Input, double A, B, the endpoints of the change of sign interval.
//
//    Input, double T, a positive error tolerance.
//
//    Input, func_base& F, the name of a user-supplied c++ functor
//    whose zero is being sought.  The input and output
//    of F() are of type double.
//
//    Output, double ZERO, the estimated value of a zero of
//    the function F.
//
{
  double c;
  double d;
  double e;
  double fa;
  double fb;
  double fc;
  double m;
  double p;
  double q;
  double r;
  double s;
  double sa;
  double sb;
  double tol;
  //
  //  Make local copies of A and B.
  //
  sa = a;
  sb = b;
  fa = f(sa);
  fb = f(sb);

  c = sa;
  fc = fa;
  e = sb - sa;
  d = e;

  constexpr double macheps = std::numeric_limits<double>::epsilon();

  for (;;) {
    if (std::fabs(fc) < std::fabs(fb)) {
      sa = sb;
      sb = c;
      c = sa;
      fa = fb;
      fb = fc;
      fc = fa;
    }

    tol = 2.0 * macheps * std::fabs(sb) + t;
    m = 0.5 * (c - sb);

    if (std::fabs(m) <= tol || fb == 0.0) {
      break;
    }

    if (std::fabs(e) < tol || std::fabs(fa) <= std::fabs(fb)) {
      e = m;
      d = e;
    } else {
      s = fb / fa;

      if (sa == c) {
        p = 2.0 * m * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        r = fb / fc;
        p = s * (2.0 * m * q * (q - r) - (sb - sa) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }

      if (0.0 < p) {
        q = -q;
      } else {
        p = -p;
      }

      s = e;
      e = d;

      if (2.0 * p < 3.0 * m * q - std::fabs(tol * q) &&
          p < std::fabs(0.5 * s * q)) {
        d = p / q;
      } else {
        e = m;
        d = e;
      }
    }
    sa = sb;
    fa = fb;

    if (tol < std::fabs(d)) {
      sb = sb + d;
    } else if (0.0 < m) {
      sb = sb + tol;
    } else {
      sb = sb - tol;
    }

    fb = f(sb);

    if ((0.0 < fb && 0.0 < fc) || (fb <= 0.0 && fc <= 0.0)) {
      c = sa;
      fc = fa;
      e = sb - sa;
      d = e;
    }
  }
  return sb;
}
//****************************************************************************80

//****************************************************************************80

template<typename F_t>
inline double local_min ( double a, double b, double t, F_t& f,
                  double &x )

//****************************************************************************80
//
//  Purpose:
//
//    LOCAL_MIN seeks a local minimum of a function F(X) in an interval [A,B].
//
//  Discussion:
//
//    The method used is a combination of golden section search and
//    successive parabolic interpolation.  Convergence is never much slower
//    than that for a Fibonacci search.  If F has a continuous second
//    derivative which is positive at the minimum (which is not at A or
//    B), then convergence is superlinear, and usually of the order of
//    about 1.324....
//
//    The values EPS and T define a tolerance TOL = EPS * abs ( X ) + T.
//    F is never evaluated at two points closer than TOL.
//
//    If F is a unimodal function and the computed values of F are always
//    unimodal when separated by at least SQEPS * abs ( X ) + (T/3), then
//    LOCAL_MIN approximates the abscissa of the global minimum of F on the
//    interval [A,B] with an error less than 3*SQEPS*abs(LOCAL_MIN)+T.
//
//    If F is not unimodal, then LOCAL_MIN may approximate a local, but
//    perhaps non-global, minimum to the same accuracy.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 July 2011
//
//  Author:
//
//    Original FORTRAN77 version by Richard Brent.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Richard Brent,
//    Algorithms for Minimization Without Derivatives,
//    Dover, 2002,
//    ISBN: 0-486-41998-3,
//    LC: QA402.5.B74.
//
//  Parameters:
//
//    Input, double A, B, the endpoints of the interval.
//
//    Input, double T, a positive absolute error tolerance.
//
//    Input, func_base& F, a user-supplied c++ functor whose
//    local minimum is being sought.  The input and output
//    of F() are of type double.
//
//    Output, double &X, the estimated value of an abscissa
//    for which F attains a local minimum value in [A,B].
//
//    Output, double LOCAL_MIN, the value F(X).
//
{
    double c;
    double d;
    double e;
    double eps;
    double fu;
    double fv;
    double fw;
    double fx;
    double m;
    double p;
    double q;
    double r;
    double sa;
    double sb;
    double t2;
    double tol;
    double u;
    double v;
    double w;
    //
    //  C is the square of the inverse of the golden ratio.
    //
    c = 0.5 * ( 3.0 - sqrt ( 5.0 ) );

    eps = sqrt ( std::numeric_limits<double>::epsilon());

    sa = a;
    sb = b;
    x = sa + c * ( b - a );
    w = x;
    v = w;
    e = 0.0;
    fx = f ( x );
    fw = fx;
    fv = fw;

    for ( ; ; )
    {
        m = 0.5 * ( sa + sb ) ;
        tol = eps * fabs ( x ) + t;
        t2 = 2.0 * tol;
        //
        //  Check the stopping criterion.
        //
        if ( fabs ( x - m ) <= t2 - 0.5 * ( sb - sa ) )
        {
            break;
        }
        //
        //  Fit a parabola.
        //
        r = 0.0;
        q = r;
        p = q;

        if ( tol < fabs ( e ) )
        {
            r = ( x - w ) * ( fx - fv );
            q = ( x - v ) * ( fx - fw );
            p = ( x - v ) * q - ( x - w ) * r;
            q = 2.0 * ( q - r );
            if ( 0.0 < q )
            {
                p = - p;
            }
            q = fabs ( q );
            r = e;
            e = d;
        }

        if ( fabs ( p ) < fabs ( 0.5 * q * r ) &&
            q * ( sa - x ) < p &&
            p < q * ( sb - x ) )
        {
            //
            //  Take the parabolic interpolation step.
            //
            d = p / q;
            u = x + d;
            //
            //  F must not be evaluated too close to A or B.
            //
            if ( ( u - sa ) < t2 || ( sb - u ) < t2 )
            {
                if ( x < m )
                {
                    d = tol;
                }
                else
                {
                    d = - tol;
                }
            }
        }
        //
        //  A golden-section step.
        //
        else
        {
            if ( x < m )
            {
                e = sb - x;
            }
            else
            {
                e = sa - x;
            }
            d = c * e;
        }
        //
        //  F must not be evaluated too close to X.
        //
        if ( tol <= fabs ( d ) )
        {
            u = x + d;
        }
        else if ( 0.0 < d )
        {
            u = x + tol;
        }
        else
        {
            u = x - tol;
        }

        fu = f ( u );
        //
        //  Update A, B, V, W, and X.
        //
        if ( fu <= fx )
        {
            if ( u < x )
            {
                sb = x;
            }
            else
            {
                sa = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        }
        else
        {
            if ( u < x )
            {
                sa = u;
            }
            else
            {
                sb = u;
            }

            if ( fu <= fw || w == x )
            {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            }
            else if ( fu <= fv || v == x || v == w )
            {
                v = u;
                fv = fu;
            }
        }
    }
    return fx;
}
//****************************************************************************80

//****************************************************************************80
template<typename F_t>
double glomin ( double a, double b, double c, double m, double e, double t,
               F_t& f, double &x )

//****************************************************************************80
//
//  Purpose:
//
//    GLOMIN seeks a global minimum of a function F(X) in an interval [A,B].
//
//  Discussion:
//
//    This function assumes that F(X) is twice continuously differentiable
//    over [A,B] and that F''(X) <= M for all X in [A,B].
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 July 2011
//
//  Author:
//
//    Original FORTRAN77 version by Richard Brent.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Richard Brent,
//    Algorithms for Minimization Without Derivatives,
//    Dover, 2002,
//    ISBN: 0-486-41998-3,
//    LC: QA402.5.B74.
//
//  Parameters:
//
//    Input, double A, B, the endpoints of the interval.
//    It must be the case that A < B.
//
//    Input, double C, an initial guess for the global
//    minimizer.  If no good guess is known, C = A or B is acceptable.
//
//    Input, double M, the bound on the second derivative.
//
//    Input, double E, a positive tolerance, a bound for the
//    absolute error in the evaluation of F(X) for any X in [A,B].
//
//    Input, double T, a positive error tolerance.
//
//    Input, func_base& F, a user-supplied c++ functor whose
//    global minimum is being sought.  The input and output
//    of F() are of type double.
//
//    Output, double &X, the estimated value of the abscissa
//    for which F attains its global minimum value in [A,B].
//
//    Output, double GLOMIN, the value F(X).
//
{
    double a0;
    double a2;
    double a3;
    double d0;
    double d1;
    double d2;
    double h;
    int k;
    double m2;
//    double macheps;
    double p;
    double q;
    double qs;
    double r;
    double s;
    double sc;
    double y;
    double y0;
    double y1;
    double y2;
    double y3;
    double yb;
    double z0;
    double z1;
    double z2;

    a0 = b;
    x = a0;
    a2 = a;
    y0 = f ( b );
    yb = y0;
    y2 = f ( a );
    y = y2;

    if ( y0 < y )
    {
        y = y0;
    }
    else
    {
        x = a;
    }

    if ( m <= 0.0 || b <= a )
    {
        return y;
    }

    constexpr double macheps = std::numeric_limits<double>::epsilon();

    m2 = 0.5 * ( 1.0 + 16.0 * macheps ) * m;

    if ( c <= a || b <= c )
    {
        sc = 0.5 * ( a + b );
    }
    else
    {
        sc = c;
    }

    y1 = f ( sc );
    k = 3;
    d0 = a2 - sc;
    h = 9.0 / 11.0;

    if ( y1 < y )
    {
        x = sc;
        y = y1;
    }
    //
    //  Loop.
    //
    for ( ; ; )
    {
        d1 = a2 - a0;
        d2 = sc - a0;
        z2 = b - a2;
        z0 = y2 - y1;
        z1 = y2 - y0;
        r = d1 * d1 * z0 - d0 * d0 * z1;
        p = r;
        qs = 2.0 * ( d0 * z1 - d1 * z0 );
        q = qs;

        if ( k < 1000000 || y2 <= y )
        {
            for ( ; ; )
            {
                if ( q * ( r * ( yb - y2 ) + z2 * q * ( ( y2 - y ) + t ) ) <
                    z2 * m2 * r * ( z2 * q - r ) )
                {
                    a3 = a2 + r / q;
                    y3 = f ( a3 );

                    if ( y3 < y )
                    {
                        x = a3;
                        y = y3;
                    }
                }
                k = ( ( 1611 * k ) % 1048576 );
                q = 1.0;
                r = ( b - a ) * 0.00001 * ( double ) ( k );

                if ( z2 <= r )
                {
                    break;
                }
            }
        }
        else
        {
            k = ( ( 1611 * k ) % 1048576 );
            q = 1.0;
            r = ( b - a ) * 0.00001 * ( double ) ( k );

            while ( r < z2 )
            {
                if ( q * ( r * ( yb - y2 ) + z2 * q * ( ( y2 - y ) + t ) ) <
                    z2 * m2 * r * ( z2 * q - r ) )
                {
                    a3 = a2 + r / q;
                    y3 = f ( a3 );

                    if ( y3 < y )
                    {
                        x = a3;
                        y = y3;
                    }
                }
                k = ( ( 1611 * k ) % 1048576 );
                q = 1.0;
                r = ( b - a ) * 0.00001 * ( double ) ( k );
            }
        }

        r = m2 * d0 * d1 * d2;
        s = sqrt ( ( ( y2 - y ) + t ) / m2 );
        h = 0.5 * ( 1.0 + h );
        p = h * ( p + 2.0 * r * s );
        q = q + 0.5 * qs;
        r = - 0.5 * ( d0 + ( z0 + 2.01 * e ) / ( d0 * m2 ) );

        if ( r < s || d0 < 0.0 )
        {
            r = a2 + s;
        }
        else
        {
            r = a2 + r;
        }

        if ( 0.0 < p * q )
        {
            a3 = a2 + p / q;
        }
        else
        {
            a3 = r;
        }

        for ( ; ; )
        {
            a3 = std::max(a3,r); //r8_max ( a3, r );

            if ( b <= a3 )
            {
                a3 = b;
                y3 = yb;
            }
            else
            {
                y3 = f ( a3 );
            }

            if ( y3 < y )
            {
                x = a3;
                y = y3;
            }

            d0 = a3 - a2;

            if ( a3 <= r )
            {
                break;
            }

            p = 2.0 * ( y2 - y3 ) / ( m * d0 );

            if ( ( 1.0 + 9.0 * macheps ) * d0 <= fabs ( p ) )
            {
                break;
            }

            if ( 0.5 * m2 * ( d0 * d0 + p * p ) <= ( y2 - y ) + ( y3 - y ) + 2.0 * t )
            {
                break;
            }
            a3 = 0.5 * ( a2 + a3 );
            h = 0.9 * h;
        }

        if ( b <= a3 )
        {
            break;
        }

        a0 = sc;
        sc = a2;
        a2 = a3;
        y0 = y1;
        y1 = y2;
        y2 = y3;
    }

    return y;
}
//****************************************************************************80
