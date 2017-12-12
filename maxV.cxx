// clang++ -std=c++14 -O3 -ffast-math -dynamiclib -o maxV.dylib maxV.cxx -I eigen-eigen-67e894c6cd8f/

// Implements the linear bipartite ranking algorithm described in
// section 17.5.1 of the book "Shalev-Shwartz and
// Ben-David. Understanding Machine Learning: From Theory to
// Algorithms."


#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

//#define EIGEN_NO_DEBUG
#include <Eigen/Core>

extern "C" float fit(unsigned int d, unsigned int N, float *x, float *y,
		     unsigned int nofs, size_t *offsets, size_t *counts,
		     unsigned int T, float alpha);

using namespace Eigen;

template <typename TA, typename TB,
	  typename real=typename TA::RealScalar>
real delta(const MatrixBase<TA>& y1, const MatrixBase<TB>& y)
{
    real a = ((y.array() >= 0) && (y1.array() >= 0)).count();
    real b = ((y.array() < 0) && (y1.array() >= 0)).count();
    real c = ((y.array() >= 0) && (y1.array() < 0)).count();
    if(a == 0) {
	if(b + c == 0) {
	    return 0;
	} else {
	    return 1;
	}
    }
    real del = (1 - (2*a)/(2*a + b + c));
    /*
    std::cout << "y1 = " << y1 << ", y = " << y << "\n";
    std::cout << ((y.array() >= 0) && (y1.array() >= 0)) << "\n";
    std::cout << "a,b,c,del = (" << a << ", " << b << ", " << c << ", " << del << ")\n";
    */
    return del; // F1 score
}

template <typename TA, typename TB, typename TC, typename TD,
	  typename real=typename TA::Scalar>
real loss1(const MatrixBase<TA>& x, const MatrixBase<TB>& y,
	   const MatrixBase<TC>& w, const MatrixBase<TD>& v)
{
    return delta(v, y) + (v-y).dot(w*x);
    auto mu = w*x;
    real d = delta(v, y);
    real alpha = d + v.dot(mu);
//    std::cout << "v = " << v << ", delta = " << d << "\n";
    return alpha;
    
//    return alpha - y.dot(w*x);

}

Matrix<float, 1, 8> char2label(char c)
{
    Matrix<float, 1, 8> v;
    for(unsigned int i=0; i < 8; i++) {
	v(i) = c & (1 << i) ? 1 : -1;
    }
    return v;
}

template <typename TA, typename TB, typename TC,
	  typename scalar=typename TA::Scalar>
Matrix<scalar, 1, TB::ColsAtCompileTime> maxV(const MatrixBase<TA>& x,
				const MatrixBase<TB>& y,
				const MatrixBase<TC>& w)
{
    const size_t r = y.cols();
    std::vector<size_t> idx(r);
    auto p = idx.begin();
    auto n = idx.end()-1;
    for(unsigned int k=0; k < r; k++) {
	if(y(0, k) >= 0) {
	    *p++ = k;
	} else {
	    *n-- = k;
	}
    }
    unsigned int P = p - idx.begin();
    unsigned int N = r - P;
    Matrix<scalar, 1, TA::ColsAtCompileTime> mu = w*x;
    std::sort(idx.begin(), p, [&mu](size_t a, size_t b) { return mu(b) < mu(a); });
    std::sort(p, idx.end(), [&mu](size_t a, size_t b) { return mu(b) < mu(a); });
    Matrix<scalar, 1, TB::ColsAtCompileTime> v_star;
    v_star.resizeLike(mu);
    scalar alpha_star = -std::numeric_limits<scalar>::infinity();
    for(unsigned int a = 0; a <= P; a++) {
	unsigned int c = P - a;
	for(unsigned int b = 0; b <= N; b++) {
	    unsigned int d = N - b;
	    Matrix<scalar, 1, TB::ColsAtCompileTime> v;
	    v.resizeLike(mu);
	    for(unsigned int k=0; k < a; k++) {
		v(idx[k]) = 1;
	    }
	    for(unsigned int k=a; k < P; k++) {
		v(idx[k]) = -1;
	    }
	    for(unsigned int k=0; k < b; k++) {
		v(idx[P+k]) = 1;
	    }
	    for(unsigned int k=b; k < N; k++) {
		v(idx[P+k]) = -1;
	    }
	    scalar delta = (a != 0) ? (1.0 - (2.0*(scalar)a)/(2.0*(scalar)a + (scalar)b + (scalar)c)) : 1;
	    scalar alpha = delta + v.dot(mu);
//	    std::cout << "v = " << v << ", a = " << a << ", b = " << b << ", delta = " << delta << ", alpha = " << alpha << ", alpha* = " << alpha_star << "\n";
	    if(alpha > alpha_star) {
		alpha_star = alpha;
		v_star = v;
	    }
	}
    }	    
    return v_star;
}

unsigned long maxVrand(unsigned int r, unsigned int d, unsigned int iter)
{
    MatrixXf x = MatrixXf::Random(d, r);
    MatrixXf y = MatrixXf::Random(1, r).cwiseSign();
    MatrixXf w = MatrixXf::Random(1, d);
    auto start = std::chrono::steady_clock::now();
    for(unsigned int i=0; i < iter; i++) {
	auto v_star = maxV(x, y, w);
    }	
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> 
	(std::chrono::steady_clock::now() - start);
    return duration.count();
}

template <typename TA, typename TB, typename TC,
	  typename scalar=typename TA::Scalar>
Matrix<scalar, 1, TC::ColsAtCompileTime> grad1(const MatrixBase<TA>& x,
					       const MatrixBase<TB>& y,
					       const MatrixBase<TC>& w)
{
    Matrix<scalar, 1, TB::ColsAtCompileTime> v = maxV(x, y, w);
//    std::cout << "v: " << v << " y: " << y << "\n";
//    std::cout << "v-y: " << v-y << " ";
    Matrix<scalar, 1, TC::ColsAtCompileTime> r = (v-y)*x.transpose() / x.cols();
#if 0
    if(r.norm() > 10000) {
	std::cout << "r.norm = " << r.norm() << "\n";
	std::cout << "v: " << v << "\n";
	std::cout << "y: " << y << "\n";
	std::cout << "v-y: " << v-y << "\n";
    }
#endif
    return r;
}

// xoroshiro128+ PRNG
uint64_t s[2] = {2, 2};

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

uint64_t xoroshiro128(void) {
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	s[1] = rotl(s1, 36); // c

	return result;
}

template <typename TA, typename TB, typename TC,
	  typename scalar=typename TA::Scalar>
scalar eval(const MatrixBase<TA>& x,
	    const MatrixBase<TB>& y,
	    const MatrixBase<TC>& w,
	    unsigned int r)
{
    scalar l = 0.0;
    unsigned int N = x.cols();
    unsigned int d = x.rows();
    for(unsigned int i=0; i < N / r; i++) {
	auto x_i = x.block(0, i*r, d, r);
	auto y_i = y.block(0, i*r, 1, r);
	auto y1 = w*x_i;
	l += delta(y1, y_i);
	/*
	std::cout << "eval(" << x_i.transpose() << ", "
		  << y_i << ", " << w << ") " << "\n";
	std::cout << "delta(" << y1 << ", " << y_i << ") = " << delta(y1, y_i) << "\n";
	*/
    }
    return l / (N/r);
}

template <typename TA, typename TB, typename TC,
	  typename scalar=typename TA::Scalar>
scalar eval(const MatrixBase<TA>& x,
	    const MatrixBase<TB>& y,
	    const MatrixBase<TC>& w,
	    unsigned int nofs, size_t *offsets, size_t *counts, bool disp=false)
{
    scalar l = 0.0;
    unsigned int N = x.cols();
    unsigned int d = x.rows();
    for(unsigned int i=0; i < nofs; i++) {
	auto x_i = x.block(0, offsets[i], d, counts[i]);
	auto y_i = y.block(0, offsets[i], 1, counts[i]);
	auto y1 = w*x_i;
	if(disp) {
	    std::cout << "f1(" << i << ") = " << delta(y1, y_i) << "\n";
//	    std::cout << "  y: " << y_i << "\n  y1: " << y1 << "\n";
	}
	l += delta(y1, y_i);
	/*
	std::cout << "eval(" << x_i.transpose() << ", "
		  << y_i << ", " << w << ") " << "\n";
	std::cout << "delta(" << y1 << ", " << y_i << ") = " << delta(y1, y_i) << "\n";
	*/
    }
    return l / nofs;
}


float fit(unsigned int d, unsigned int N, float *x, float *y,
	  unsigned int nofs, size_t *offsets, size_t *counts,
	  unsigned int T, float alpha)
{
    Map<Matrix<float, Dynamic, Dynamic, RowMajor> > x_train(x, d, N);
    Map<Matrix<float, Dynamic, Dynamic, RowMajor> > y_train(y, 1, N);
    MatrixXf w(1, d);
    MatrixXf wcum(1, d);
    wcum.setZero();
    w.setRandom();
//    w(0, 0) = 1;
//    w(0, 1) = -1;
//    w.setRandom();
    /*
    constexpr unsigned int T = 100;
    constexpr float alpha = 1;
    */
//    std::cout << "y: " << y_train << "\n";
    std::cout << "init w: " << w << '\n';
    std::cout << "init l: " << eval(x_train, y_train, w, nofs, offsets, counts, false) << "\n";
    for(unsigned int t=0; t < T; t++) {
	unsigned long ii = xoroshiro128();
//	unsigned long ii = t;
	unsigned int i = ii % nofs;
	auto x_i = x_train.block(0, offsets[i], d, counts[i]);
	auto y_i = y_train.block(0, offsets[i], 1, counts[i]);
	Matrix<float, 1, Dynamic> dw = -(alpha/(1.0+t*1e-2)) * grad1(x_i, y_i, w);
	/*
	std::cout << "x: " << x_i.transpose() << " y: " << y_i << " w: " << w 
		  << " dw: " << dw << '\n';
	*/
//	std::cout << "t: " << t << " i: " << i << " dw: " << dw << " w: " << w << " -w1/w0: " << -w(2)/w(1) << "\n";
	if(t % 1000 == 0 || t == T-1) {
	    std::cout << "t: " << t << " l: " << eval(x_train, y_train, w, nofs, offsets, counts, false) /*<< " w: " << w */ << "\n";
	}
	w += dw;
	wcum += w;
    }
    wcum /= (float)T;
    std::cout "cumul: w = " << wcum << '\n';
    std::cout << "cumul: l = " << eval(x_train, y_train, wcum, nofs, offsets, counts, false) << "\n";
}

#if 0
int main()
{
    constexpr unsigned int N = 100000;
    constexpr unsigned int r = 100;
    constexpr unsigned int d = 50;
    MatrixXf x_train(d, N);
    MatrixXf x_test(d, N);
    x_train.setRandom();
    x_test.setRandom();
    MatrixXf w_true(1, d);
    w_true.setRandom();
    MatrixXf y_train = (w_true*x_train).cwiseSign();
    MatrixXf y_test = (w_true*x_test).cwiseSign();
//    std::cout << x_train << "\n";
//    std::cout << y_train << "\n";
    MatrixXf w(1, d);
//    w.setZero();
    w.setRandom();
    constexpr unsigned int T = 100;
    constexpr float alpha = 1;
    std::cout << "delta_true: " << delta(w_true*x_train, y_train) << "\n";
    std::cout << "l_true: " << eval(x_train, y_train, w_true, r) << "\n";
    std::cout << w << '\n';
    for(unsigned int t=0; t < T; t++) {
	unsigned long ii = xoroshiro128();
	unsigned int i = ii % (N/r);
	auto x_i = x_train.block<d, r>(0, i*r);
	auto y_i = y_train.block<1, r>(0, i*r);
	auto dw = -alpha * grad1(x_i, y_i, w);
	/*
	std::cout << "x: " << x_i.transpose() << " y: " << y_i << " w: " << w 
		  << " dw: " << dw << '\n';
	*/
	std::cout << "l: " << eval(x_test, y_test, w, r) << " |grad|: " << dw.norm() << "\n";
	w += dw;
    }
}
#endif
#if 0
int main()
{
    std::cout << "d, r, t, iter/s\n";
    float iter = 1e4;
    for(unsigned int d=1; d <= 128; d*=2) {
	for(unsigned int r=1; r <= 128; r*=2) {
	    unsigned long t = maxVrand(r, d, 1e4);
	    std::cout << d << ", " << r << ", " << t << ", " << iter/t * 1e3 << "\n";
	}
    }
    return 0;
}
#endif

#if 0
int main()
{
    constexpr int r = 8;
    constexpr int d = 10;
    Matrix<float, d, r> x = Matrix<float, d, r>::Random();
    Matrix<float, 1, r> y = Matrix<float, 1, r>::Random().cwiseSign();
    Matrix<float, 1, d> w = Matrix<float, 1, d>::Random();
    std::cout << "x = " << x << ", y = " << y << ", w = " << w << "\n";
    auto v_star = maxV(x, y, w);
    float l_star = loss1(x, y, w, v_star);
    std::cout << "l* = " << l_star << ", v* = " << v_star << '\n';
    for(unsigned int c=0; c < (1 << r); c++) {
	auto v = char2label(c).head<r>();
	auto l = loss1(x, y, w, v);
//	std::cout << "l = " << l << ", v = " << v << '\n';
	if(l > l_star) {
	    std::cout << "Found larger: "
		      << "l = " << l << ", v = " << v << '\n';
	}
    }
    return 0;
}
#endif
