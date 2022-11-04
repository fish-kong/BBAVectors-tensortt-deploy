#ifndef POLYIOU_POLYIOU_H
#define POLYIOU_POLYIOU_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>
using namespace std;
using namespace cv;

#define maxn 51
const double eps = 1E-8;
int sig(double d);
//struct Point {
//	double x, y; Point() {}
//	Point(double x, double y) :x(x), y(y) {}
//	bool operator==(const Point&p)const {
//		return sig(x - p.x) == 0 && sig(y - p.y) == 0;
//	}
//};

double cross(Point o, Point a, Point b);
double area(Point* ps, int n);
int lineCross(Point a, Point b, Point c, Point d, Point&p);
void polygon_cut(Point*p, int&n, Point a, Point b, Point* pp);
double intersectArea0(Point a, Point b, Point c, Point d);
double intersectArea(Point*ps1, int n1, Point*ps2, int n2);

double iou_poly(float p[8], float q[8]);
#endif //POLYIOU_POLYIOU_H
