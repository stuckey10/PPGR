TEMPLATE = app
CONFIG += console c++17
CONFIG -= app_bundle
CONFIG -= qt
QMAKE_CXXFLAGS += -DARMA_DONT_USE_WRAPPER
LIBS += -lopenblas -llapack -lm -lglut -lGLU -lGL

SOURCES += \
        main.cpp
