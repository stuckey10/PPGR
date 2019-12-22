#include <iostream>
#include <armadillo>
#include <cmath>
#include <tuple>
#include <GL/glut.h>

static double alpha1 = M_PI/3, beta1 = M_PI / 4, gama1 = M_PI/5;
static double alpha2 = -M_PI/5, beta2 = M_PI / 2, gama2 = M_PI/3;

static double x1 = -3, _y1 = -3, z1 = 7;
static double x2 = 3, y2 = 5, z2 = -8;

static double xt = 0, yt = 0, zt = 0;

static std::tuple<double, double, double, double> qt;

static std::tuple<double, double, double, double> q1;
static std::tuple<double, double, double, double> q2;

static double tm = 10;
static double t = 0;

static int activeTimer = 0;

static arma::Mat<double> A(3,3);
static double phi;

static void onTimer(int value);
static void onKeyboard(unsigned char key, int x, int y);
static void onDisplay();

static void quartenions();

arma::Mat<double> Euler2A(double phi, double theta, double ksi);
std::tuple<arma::Row<double>, double> AxisAngle(arma::Mat<double> A);
arma::Mat<double> Rodrigez(arma::colvec p, double phi);
std::tuple<double, double, double> A2Euler (arma::Mat<double> A);
std::tuple<double, double, double, double> AxisAngle2Q(arma::Row<double> p, double phi);
std::tuple<arma::Row<double>, double> Q2AxisAngle(std::tuple<double, double, double, double>);

double round_up(double value, int decimal_places);


int main(int argc, char* argv[]){

    quartenions();

    double phi = acos(std::get<0>(q1)*std::get<0>(q2) +
                      std::get<1>(q1)*std::get<1>(q2) +
                      std::get<2>(q1)*std::get<2>(q2) +
                      std::get<3>(q1)*std::get<3>(q2));

    std::cout << phi << std::endl;
    if(phi > M_PI/2 || phi < -M_PI/2){
        q1 = {-std::get<0>(q1), -std::get<1>(q1), -std::get<2>(q1), -std::get<3>(q1)};
        phi = acos(std::get<0>(q1)*std::get<0>(q2) +
                   std::get<1>(q1)*std::get<1>(q2) +
                   std::get<2>(q1)*std::get<2>(q2) +
                   std::get<3>(q1)*std::get<3>(q2));
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);

    glutInitWindowPosition(100, 80);
    glutInitWindowSize(800, 500);
    glutCreateWindow("SLER-PPGR");

    glutKeyboardFunc(onKeyboard);
    glutDisplayFunc(onDisplay);

    glClearColor(static_cast<GLclampf>(0.1),
                 static_cast<GLclampf>(0.1),
                 static_cast<GLclampf>(0.1),
                 0);
    glEnable(GL_DEPTH_TEST);

    glLineWidth(1);

    glutMainLoop();

    return 0;
}

arma::Mat<double> Euler2A(double phi, double theta, double ksi){
    arma::Mat<double> A = {{1, 0, 0},
                           {0, cos(phi), -sin(phi)},
                           {0, sin(phi), cos(phi)}},

                      B = {{cos(theta), 0, sin(theta)},
                           {0, 1, 0},
                           {-sin(theta), 0, cos(theta)}},

                      C = {{cos(ksi), -sin(ksi), 0},
                           {sin(ksi), cos(ksi), 0},
                           {0,0,1}};

        return C*B*A;
}

std::tuple<arma::Row<double>, double> AxisAngle(arma::Mat<double> A){

    if(!(fabs(arma::det(A*A.t())-1.0) < 0.0001)){
        std::cout << det(A*A.t()) << std::endl;
        std::cout << "Determinanta nije 1" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    arma::cx_vec eigval;
    arma::cx_mat eigvec;

    arma::eig_gen(eigval, eigvec, A);

    arma::uword p_value = 0;

    for(int i = 0; i < 3; i++){
        if(fabs(std::real(eigval[static_cast<arma::uword>(i)])-1.0) < 0.001){
            p_value = static_cast<arma::uword>(i);
            break;
        }
    }

    arma::Row<double> vec = {std::real(eigvec[3*p_value]),
                             std::real(eigvec[3*p_value+1]),
                             std::real(eigvec[3*p_value+2])};
    auto p = vec;

    auto len = std::inner_product(std::begin(vec), std::end(vec), std::begin(vec), 0.0);

    auto vecLen = 1/sqrt(len);
    p = p*vecLen;

    arma::Row<double> u = {-p[1], p[0], 0};
    u = u * (1/sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]));


    std::transform(std::begin(p), std::end(p), std::begin(p), [vecLen](auto elem){return vecLen*elem;});
    arma::mat up = A*u.t();

    double phi = acos(u[0]*up[0] + u[1]*up[1] + u[2]*up[2]);

    arma::Mat<double> B = {{u[0], u[1], u[2]}, {up[0], up[1], up[2]}, {p[0], p[1], p[2]}};
    if( arma::det(B) < 0)
        p = -p;

    return std::make_tuple(p, phi);
}

arma::Mat<double> Rodrigez(arma::colvec p, double phi){
    p = p/sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);

    arma::Mat<double> ppt = p*p.t();

    arma::Mat<double> E_ppt = arma::eye<arma::Mat<double>>(3,3) - ppt;

    arma::Mat<double> px = {{0, -p[2], p[1]},
                            {p[2], 0, -p[0]},
                            {-p[1], p[0], 0}};


    arma::Mat<double> retval = ppt + (cos(phi) * E_ppt) + (sin(phi) * px);
    return retval;
}

std::tuple<double, double, double> A2Euler (arma::Mat<double> A){
    double ksi, teta, phi;

    if(!(fabs(arma::det(A)-1.0) < 0.0001)){
        std::cout << "Determinanta nije 1" << std::endl;
        exit(EXIT_FAILURE);
    }

    if(A(2,0) <1){
        if(A(2,0) > -1){
            ksi = atan2(A(1,0), A(0,0));
            teta = asin(-A(2,0));
            phi = atan2(A(2,1), A(2,2));
        }
        else{
            ksi = atan2(-A(0,1), A(1,1));
            teta = M_PI/2;
            phi = 0;
        }
    }
    else{
        ksi = atan2(-A(0,1), A(1,1));
        teta = -M_PI/2;
        phi = 0;
    }

    return std::make_tuple(phi, teta, ksi);
}

std::tuple<double, double, double, double> AxisAngle2Q(arma::Row<double> p, double phi){
    double w = cos(phi/2);

    double sum = 0;

    for(auto a : p){
        sum += a*a;
    }

    sum = sqrt(sum);

    p = (1/sum)*p;

    p = sin(phi/2)*p;

    return std::make_tuple(p[0], p[1], p[2], w);
}

std::tuple<arma::Row<double>, double> Q2AxisAngle(std::tuple<double, double, double, double> q){
    double q0 = std::get<0>(q);
    double q1 = std::get<1>(q);
    double q2 = std::get<2>(q);
    double q3 = std::get<3>(q);

    std::vector<double> qs{q0, q1, q2, q3};
    arma::Row<double> p{0,0,0};

    if(q3 < 0){
        std::transform(qs.begin(), qs.end(), qs.begin(), [](auto elem){return -elem;});
    }

    double phi = 2*std::acos(qs[3]);

    if(fabs(qs[3]) == 1){
        p = {1,0,0};
    }
    else{
        double norm = sqrt(q0*q0+q1*q1+q2*q2);
        p = {qs[0], qs[1], qs[2]};
        std::transform(p.begin(), p.end(), p.begin(), [norm](auto &elem){return elem/norm;});
    }

    return std::make_tuple(p, phi);
}

double round_up(double value, int decimal_places) {
    const double multiplier = std::pow(10.0, decimal_places);
    return std::ceil(value * multiplier) / multiplier;
}

void SLERP(){

    if(phi < M_PI/12 && phi > -M_PI/12){
        auto q11 = (1-t/tm)*std::get<0>(q1),
             q21 = (1-t/tm)*std::get<1>(q1),
             q31 = (1-t/tm)*std::get<2>(q1),
             q41 = (1-t/tm)*std::get<3>(q1);

        auto w11 = (t/tm)*std::get<0>(q2),
             w21 = (t/tm)*std::get<1>(q2),
             w31 = (t/tm)*std::get<2>(q2),
             w41 = (t/tm)*std::get<3>(q2);


        qt = {q11+w11, q21+w21, q31+w31, q41+w41};
        double len = std::get<0>(qt)*std::get<0>(qt) +
                     std::get<1>(qt)*std::get<1>(qt) +
                     std::get<2>(qt)*std::get<2>(qt) +
                     std::get<3>(qt)*std::get<3>(qt);

        len = sqrt(len);

        qt = {std::get<0>(qt)/len, std::get<1>(qt)/len, std::get<2>(qt)/len, std::get<3>(qt)/len};
    }
    else{
        auto q11 = sin(phi*(1 - t/tm))/sin(phi)*std::get<0>(q1),
             q21 = sin(phi*(1 - t/tm))/sin(phi)*std::get<1>(q1),
             q31 = sin(phi*(1 - t/tm))/sin(phi)*std::get<2>(q1),
             q41 = sin(phi*(1 - t/tm))/sin(phi)*std::get<3>(q1);

        auto w11 = sin(phi*(t/tm))/sin(phi)*std::get<0>(q2),
             w21 = sin(phi*(t/tm))/sin(phi)*std::get<1>(q2),
             w31 = sin(phi*(t/tm))/sin(phi)*std::get<2>(q2),
             w41 = sin(phi*(t/tm))/sin(phi)*std::get<3>(q2);

        qt = {q11+w11, q21+w21, q31+w31, q41+w41};
    }

    xt = (1 - t/tm)*x1 + (t/tm)*x2;
    yt = (1 - t/tm)*_y1 + (t/tm)*y2;
    zt = (1 - t/tm)*z1 + (t/tm)*z2;

    return;
}

void drawCube(){

    glColor3f(51.0f/255, 255.0f, 255.0f);
    glutWireCube(2);

    glColor3f(255, 51.0f/255, 255.0f);
    glutWireOctahedron();
}

void beginEndPostiotion(){
    arma::Mat<double> E2A1(3,3);

    E2A1 = Euler2A(alpha1, beta1, gama1);

    GLdouble matrix1[16] = {E2A1(0, 0), E2A1(1, 0), E2A1(2, 0), 0,
                            E2A1(0, 1), E2A1(1, 1), E2A1(2, 1), 0,
                            E2A1(0, 2), E2A1(1, 2), E2A1(2, 2), 0,
                            x1, _y1, z1, 1 };

    glPushMatrix();

    glMultMatrixd(matrix1);
    drawCube();

    glPopMatrix();

    arma::Mat<double> E2A2(3,3);
    E2A2 = Euler2A(alpha2, beta2, gama2);

    GLdouble matrix2[16] = {E2A2(0, 0), E2A2(1, 0), E2A2(2, 0), 0,
                            E2A2(0, 1), E2A2(1, 1), E2A2(2, 1), 0,
                            E2A2(0, 2), E2A2(1, 2), E2A2(2, 2), 0,
                            x2, y2, z2, 1 };

    glPushMatrix();

    glMultMatrixd(matrix2);

    drawCube();

    glPopMatrix();
}

void quartenions(){
    //q1//
    arma::Mat<double> E2A1(3,3);

    E2A1 = Euler2A(alpha1, beta1, gama1);

    auto pair = AxisAngle(E2A1);

    arma::Row<double> p1 = std::get<0>(pair);

    double angle1 = std::get<1>(pair);

    q1 = AxisAngle2Q(p1, angle1);

    // q2 //

    arma::Mat<double> E2A2(3,3);

    E2A2 = Euler2A(alpha2, beta2, gama2);

    pair = AxisAngle(E2A2);

    arma::Row<double> p2 = std::get<0>(pair);

    double angle2 = std::get<1>(pair);

    q2 = AxisAngle2Q(p2, angle2);

    return;
}

static void onKeyboard(unsigned char key, int x, int y){
    std::cout << x << "-" << y << std::endl;
    switch(key){
        case 27:
            exit(0);
        case 's':
        case 'S':
            if (!activeTimer) {
                glutTimerFunc(50, onTimer, 0);
                activeTimer = 1;
            }
            break;
    }
}

static void onTimer(int value)
{
    if (value != 0)
        return;

    t += 0.1;

    if(t >= tm){
        t = 0;
        activeTimer = 0;
        glutPostRedisplay();
        return;
    }

    glutPostRedisplay();

    if (activeTimer)
        glutTimerFunc(50, onTimer, 0);
}

static void onDisplay(){

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glViewport(0, 0, 800, 500);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(50, static_cast<GLdouble>(800) / 500, 1, 00);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(15, 15, 10,
               0, 0, 0,
               0, 1, 0);


    beginEndPostiotion();

    glPushMatrix();

    SLERP();

    auto pair = Q2AxisAngle(qt);

    arma::Row<double> vekP = std::get<0>(pair);
    arma::colvec p = {vekP(0), vekP(1), vekP(2)};
    double angle = std::get<1>(pair);
    A = Rodrigez(p, angle);
    GLdouble matrix[16] = {
                           A(0, 0), A(1, 0), A(2, 0), 0,
                           A(0, 1), A(1, 1), A(2, 1), 0,
                           A(0, 2), A(1, 2), A(2, 2), 0,
                           xt,     yt,     zt, 1
                          };

    glMultMatrixd(matrix);

    drawCube();

    glPopMatrix();

    glColor3f(0.25, 0.25, 1);

    glBegin(GL_LINES);
        glColor3f(1, 0, 0);
        glVertex3d(0, 0, 0);
        glVertex3d(200, 0, 0);

        glColor3f(0, 1, 0);
        glVertex3d(0, 0, 0);
        glVertex3d(0, 200, 0);

        glColor3f(0, 0, 1);
        glVertex3d(0, 0, 0);
        glVertex3d(0, 0, 200);
    glEnd();

    glutSwapBuffers();

}
