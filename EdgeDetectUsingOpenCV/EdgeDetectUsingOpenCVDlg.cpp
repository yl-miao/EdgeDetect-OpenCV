
// EdgeDetectUsingOpenCVDlg.cpp: 实现文件
//

#include "pch.h"
#include "framework.h"
#include "EdgeDetectUsingOpenCV.h"
#include "EdgeDetectUsingOpenCVDlg.h"
#include "afxdialogex.h"

#include <opencv2\opencv.hpp>
#include <opencv2\ximgproc.hpp>

#include<opencv2\highgui\highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;


#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CEdgeDetectUsingOpenCVDlg 对话框



CEdgeDetectUsingOpenCVDlg::CEdgeDetectUsingOpenCVDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_EDGEDETECTUSINGOPENCV_DIALOG, pParent)
	, m_c(FALSE)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CEdgeDetectUsingOpenCVDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Check(pDX, IDC_CHECK1, m_c);
}

BEGIN_MESSAGE_MAP(CEdgeDetectUsingOpenCVDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CEdgeDetectUsingOpenCVDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CEdgeDetectUsingOpenCVDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CEdgeDetectUsingOpenCVDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CEdgeDetectUsingOpenCVDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CEdgeDetectUsingOpenCVDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CEdgeDetectUsingOpenCVDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CEdgeDetectUsingOpenCVDlg::OnBnClickedButton7)
END_MESSAGE_MAP()


// CEdgeDetectUsingOpenCVDlg 消息处理程序

BOOL CEdgeDetectUsingOpenCVDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 将“关于...”菜单项添加到系统菜单中。

	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}

void CEdgeDetectUsingOpenCVDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CEdgeDetectUsingOpenCVDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CEdgeDetectUsingOpenCVDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CEdgeDetectUsingOpenCVDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("D:\\sub4.tif");
	
	namedWindow("原始图", CV_WINDOW_NORMAL);
	resizeWindow("原始图", 480, 480);
	moveWindow("原始图", 5, 5);
	imshow("原始图", img);
	Mat DstPic, edge, grayImage;

	//创建与src同类型和同大小的矩阵
	DstPic.create(img.size(), img.type());

	//将原始图转化为灰度图
	cvtColor(img, grayImage, COLOR_BGR2GRAY);

	//先使用3*3内核来降噪
	blur(grayImage, edge, Size(3, 3));

	//运行canny算子
	Canny(edge, edge, 3, 9, 3);

	namedWindow("Canny效果", CV_WINDOW_NORMAL);
	resizeWindow("Canny效果", 480, 480);
	moveWindow("Canny效果", 5, 5+480);
	imshow("Canny效果", edge);

	waitKey(0);
}


void CEdgeDetectUsingOpenCVDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("D:\\sub4.tif");

	namedWindow("原始图", CV_WINDOW_NORMAL);
	resizeWindow("原始图", 480, 480);
	moveWindow("原始图", 5, 5);
	imshow("原始图", img);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;

	
	//求x方向梯度
	Sobel(img, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//imshow("x方向soble", abs_grad_x);

	//求y方向梯度
	Sobel(img, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//imshow("y向soble", abs_grad_y);
	

	//合并梯度
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	namedWindow("整体方向soble", CV_WINDOW_NORMAL);
	resizeWindow("整体方向soble", 480, 480);
	moveWindow("整体方向soble", 5, 5 + 480);
	imshow("整体方向soble", dst);

	waitKey(0);
}


void CEdgeDetectUsingOpenCVDlg::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("D:\\sub4.tif");
	namedWindow("原始图", CV_WINDOW_NORMAL);
	resizeWindow("原始图", 480, 480);
	moveWindow("原始图", 5, 5);
	imshow("原始图", img);
	Mat gray, dst, abs_dst;
	//高斯滤波消除噪声
	GaussianBlur(img, img, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//转换为灰度图
	cvtColor(img, gray, COLOR_RGB2GRAY);
	//使用Laplace函数
	//第三个参数：目标图像深度；第四个参数：滤波器孔径尺寸；第五个参数：比例因子；第六个参数：表示结果存入目标图
	Laplacian(gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	//计算绝对值，并将结果转为8位
	convertScaleAbs(dst, abs_dst);

	namedWindow("Laplace效果", CV_WINDOW_NORMAL);
	resizeWindow("Laplace效果", 480, 480);
	moveWindow("Laplace效果", 5, 5 + 480);
	imshow("Laplace效果", abs_dst);

	waitKey(0);
}


void CEdgeDetectUsingOpenCVDlg::OnBnClickedButton4()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat img = imread("D:\\sub4.tif");

	namedWindow("原始图", CV_WINDOW_NORMAL);
	resizeWindow("原始图", 480, 480);
	moveWindow("原始图", 5, 5);
	imshow("原始图", img);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, dst;
	//求x方向梯度
	Scharr(img, grad_x, CV_16S, 1, 0,  1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	//imshow("x方向Scharr", abs_grad_x);

	//求y方向梯度
	Scharr(img, grad_y, CV_16S, 0, 1,  1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	//imshow("y向Scharr", abs_grad_y);


	//合并梯度
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
	namedWindow("整体方向Scharr", CV_WINDOW_NORMAL);
	resizeWindow("整体方向Scharr", 480, 480);
	moveWindow("整体方向Scharr", 5, 5 + 480);
	imshow("整体方向Scharr", dst);

	waitKey(0);
}

void marrEdge(const Mat src, Mat& result, int kerValue, double delta)
{
	//计算LoG算子
	Mat kernel;
	//半径
	int kerLen = kerValue / 2;
	kernel = Mat_<double>(kerValue, kerValue);
	//滑窗
	for (int i = -kerLen; i <= kerLen; i++)
	{
		for (int j = -kerLen; j <= kerLen; j++)
		{
			//生成核因子
			kernel.at<double>(i + kerLen, j + kerLen) = exp(-((pow(j, 2) + pow(i, 2)) / (pow(delta, 2) * 2)))
				* ((pow(j, 2) + pow(i, 2) - 2 * pow(delta, 2)) / (2 * pow(delta, 4)));
		}
	}
	//设置输入参数
	int kerOffset = kerValue / 2;
	Mat laplacian = (Mat_<double>(src.rows - kerOffset * 2, src.cols - kerOffset * 2));
	result = Mat::zeros(src.rows - kerOffset * 2, src.cols - kerOffset * 2, src.type());
	double sumLaplacian;
	//遍历计算卷积图像的拉普拉斯算子
	for (int i = kerOffset; i < src.rows - kerOffset; ++i)
	{
		for (int j = kerOffset; j < src.cols - kerOffset; ++j)
		{
			sumLaplacian = 0;
			for (int k = -kerOffset; k <= kerOffset; ++k)
			{
				for (int m = -kerOffset; m <= kerOffset; ++m)
				{
					//计算图像卷积
					sumLaplacian += src.at<uchar>(i + k, j + m) * kernel.at<double>(kerOffset + k, kerOffset + m);
				}
			}
			//生成拉普拉斯结果
			laplacian.at<double>(i - kerOffset, j - kerOffset) = sumLaplacian;
		}
	}
	for (int y = 1; y < result.rows - 1; ++y)
	{
		for (int x = 1; x < result.cols - 1; ++x)
		{
			result.at<uchar>(y, x) = 0;
			//领域判定
			if (laplacian.at<double>(y - 1, x) * laplacian.at<double>(y + 1, x) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y, x - 1) * laplacian.at<double>(y, x + 1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y + 1, x - 1) * laplacian.at<double>(y - 1, x + 1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y - 1, x - 1) * laplacian.at<double>(y + 1, x + 1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
		}
	}
}

void CEdgeDetectUsingOpenCVDlg::OnBnClickedButton5()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat srcImage = imread("D:\\sub4.tif");

	if (!srcImage.data)
		MessageBox(_T("出错！"));

	namedWindow("原始图", CV_WINDOW_NORMAL);
	resizeWindow("原始图", 480, 480);
	moveWindow("原始图", 5, 5);
	imshow("原始图", srcImage);

	Mat edge, srcGray;
	cvtColor(srcImage, srcGray, CV_RGB2GRAY);
	marrEdge(srcGray, edge, 9, 1.6);

	namedWindow("LOG效果", CV_WINDOW_NORMAL);
	resizeWindow("LOG效果", 480, 480);
	moveWindow("LOG效果", 5, 5 + 480);
	imshow("LOG效果", edge);
	waitKey(0);
}

Vec3b RandomColor(int value)//生成随机颜色函数
{
	value = value % 255;  //生成0~255的随机数
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}

void CEdgeDetectUsingOpenCVDlg::OnBnClickedButton6()
{
	// TODO: 在此添加控件通知处理程序代码
	Mat image = imread("D:\\sub4.tif");    //载入RGB彩色图像

	namedWindow("Source Image", CV_WINDOW_NORMAL);
	resizeWindow("Source Image", 480, 480);
	moveWindow("Source Image", 5, 5);
	imshow("Source Image", image);

	//灰度化，滤波，Canny边缘检测
	Mat imageGray;
	cvtColor(image, imageGray, CV_RGB2GRAY);//灰度转换
	GaussianBlur(imageGray, imageGray, Size(5, 5), 2);   //高斯滤波
	//imshow("Gray Image", imageGray);
	Canny(imageGray, imageGray, 80, 150);
	//imshow("Canny Image", imageGray);

	//查找轮廓
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);  //轮廓	
	Mat marks(image.size(), CV_32S);   //Opencv分水岭第二个矩阵参数
	marks = Scalar::all(0);
	int index = 0;
	int compCount = 0;
	for (; index >= 0; index = hierarchy[index][0], compCount++)
	{
		//对marks进行标记，对不同区域的轮廓进行编号，相当于设置注水点，有多少轮廓，就有多少注水点
		drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
	}

	//我们来看一下传入的矩阵marks里是什么东西
	Mat marksShows;
	convertScaleAbs(marks, marksShows);
	//imshow("marksShow", marksShows);
	//imshow("轮廓", imageContours);
	watershed(image, marks);

	//我们再来看一下分水岭算法之后的矩阵marks里是什么东西
	Mat afterWatershed;
	convertScaleAbs(marks, afterWatershed);

	namedWindow("After Watershed轮廓", CV_WINDOW_NORMAL);
	resizeWindow("After Watershed轮廓", 480, 480);
	moveWindow("After Watershed轮廓", 5, 5 + 480);
	imshow("After Watershed轮廓", afterWatershed);

	//对每一个区域进行颜色填充
	Mat PerspectiveImage = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < marks.rows; i++)
	{
		for (int j = 0; j < marks.cols; j++)
		{
			int index = marks.at<int>(i, j);
			if (marks.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else
			{
				PerspectiveImage.at<Vec3b>(i, j) = RandomColor(index);
			}
		}
	}
	//imshow("After ColorFill", PerspectiveImage);

	//分割并填充颜色的结果跟原始图像融合
	Mat wshed;
	addWeighted(image, 0.4, PerspectiveImage, 0.6, 0, wshed);

	namedWindow("After ColorFill and AddWeighted Image", CV_WINDOW_NORMAL);
	resizeWindow("After ColorFill and AddWeighted Image", 480, 480);
	moveWindow("After ColorFill and AddWeighted Image", 5+480, 5 + 480);
	imshow("After ColorFill and AddWeighted Image", wshed);

	waitKey();
}


void CEdgeDetectUsingOpenCVDlg::OnBnClickedButton7()
{
	// TODO: 在此添加控件通知处理程序代码
	Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection("D:\\model.yml.gz");

	Mat3b src = imread("D:\\sub4.tif");
	namedWindow("Source Image", CV_WINDOW_NORMAL);
	resizeWindow("Source Image", 480, 480);
	moveWindow("Source Image", 5, 5);
	imshow("Source Image", src);

	Mat3f fsrc;
	src.convertTo(fsrc, CV_32F, 1.0 / 255.0);

	Mat1f edges;
	pDollar->detectEdges(fsrc, edges);

	namedWindow("结构森林效果", CV_WINDOW_NORMAL);
	resizeWindow("结构森林效果", 480, 480);
	moveWindow("结构森林效果", 5, 5 + 480);
	imshow("结构森林效果", edges);
	UpdateData(TRUE);
	BOOL state = m_c;
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(1);
	if (state) {
		Mat tmp;
		edges.convertTo(tmp, CV_8U, 1,1);
		imwrite("tmp.png", tmp, compression_params);
	}
	waitKey();
}
