import graph_builder
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from interface.streamlit_utils import render_function

import minitorch
from minitorch import MathTest, MathTestVariable

MyModule = None
minitorch


def render_math_sandbox(use_scalar: bool = False, use_tensor: bool = False) -> None:
    st.write("## Sandbox for Math Functions")
    st.write("Visualization of the mathematical tests run on the underlying code.")

    if use_scalar:
        one, two, red = MathTestVariable._comp_testing()
    else:
        one, two, red = MathTest._comp_testing()
    f_type = st.selectbox("Function Type", ["One Arg", "Two Arg", "Reduce"])
    assert f_type is not None
    select = {"One Arg": one, "Two Arg": two, "Reduce": red}

    fn = st.selectbox("Function", select[f_type], format_func=lambda a: a[0])
    assert fn is not None
    name, _, scalar = fn
    if f_type == "One Arg":
        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x)")
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        if use_scalar:
            if use_tensor:
                ys = [scalar(minitorch.tensor([p]))[0] for p in xs]
            else:
                ys = [scalar(minitorch.Scalar(p)).data for p in xs]
        else:
            ys = [scalar(p) for p in xs]
        scatter = go.Scatter(mode="lines", x=xs, y=ys)
        fig = go.Figure(scatter)
        st.write(fig)

        if use_scalar:
            st.write("Derivative f'(x)")
            ys = []
            for x in xs:
                if use_tensor:
                    x_tens = minitorch.tensor(x, requires_grad=True)
                    out = scalar(x_tens)
                    out.backward(minitorch.tensor([1.0]))
                    assert x_tens.grad is not None
                    ys.append(x_tens.grad[0])
                else:
                    x_scal = minitorch.Scalar(x)
                    out = scalar(x_scal)
                    out.backward()
                    ys.append(x_scal.derivative)
            scatter = go.Scatter(mode="lines", x=xs, y=ys)
            fig = go.Figure(scatter)
            st.write(fig)
            G = graph_builder.GraphBuilder().run(out)  # type: ignore
            G.graph["graph"] = {"rankdir": "LR"}
            st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())

    if f_type == "Two Arg":

        st.write("### " + name)
        render_function(scalar)
        st.write("Function f(x, y)")
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        if use_scalar:
            if use_tensor:
                zs = [
                    [
                        scalar(minitorch.tensor([x]), minitorch.tensor([y]))[0]
                        for x in xs
                    ]
                    for y in ys
                ]
            else:
                zs = [
                    [scalar(minitorch.Scalar(x), minitorch.Scalar(y)).data for x in xs]
                    for y in ys
                ]
        else:
            zs = [[scalar(x, y) for x in xs] for y in ys]

        scatter = go.Surface(x=xs, y=ys, z=zs)

        fig = go.Figure(scatter)
        st.write(fig)
        if use_scalar:
            a, b = [], []
            for x in xs:
                oa, ob = [], []

                if use_tensor:
                    for y in ys:
                        x_tens = minitorch.tensor([x])
                        y_tens = minitorch.tensor([y])
                        out = scalar(x_tens, y_tens)
                        out.backward(minitorch.tensor([1]))
                        assert x_tens.grad is not None
                        assert y_tens.grad is not None
                        oa.append((x, y, x_tens.grad[0]))
                        ob.append((x, y, y_tens.grad[0]))
                else:
                    for y in ys:
                        x_scal = minitorch.Scalar(x)
                        y_scal = minitorch.Scalar(y)
                        out = scalar(x_scal, y_scal)
                        out.backward()
                        assert x_scal.derivative is not None
                        assert y_scal.derivative is not None
                        oa.append((x, y, x_scal.derivative))
                        ob.append((x, y, y_scal.derivative))
                a.append(oa)
                b.append(ob)
            st.write("Derivative f'_x(x, y)")

            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in a],
                y=[[c[1] for c in a2] for a2 in a],
                z=[[c[2] for c in a2] for a2 in a],
            )
            fig = go.Figure(scatter)
            st.write(fig)
            st.write("Derivative f'_y(x, y)")
            scatter = go.Surface(
                x=[[c[0] for c in a2] for a2 in b],
                y=[[c[1] for c in a2] for a2 in b],
                z=[[c[2] for c in a2] for a2 in b],
            )
            fig = go.Figure(scatter)
            st.write(fig)
    if f_type == "Reduce":
        st.write("### " + name)
        render_function(scalar)
        xs = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]
        ys = [((x / 1.0) - 50.0 + 1e-5) for x in range(1, 100)]

        if use_tensor:
            scatter = go.Surface(
                x=xs,
                y=ys,
                z=[[scalar(minitorch.tensor([x, y]))[0] for x in xs] for y in ys],
            )
        else:
            scatter = go.Surface(
                x=xs, y=ys, z=[[scalar([x, y]) for x in xs] for y in ys]
            )
        fig = go.Figure(scatter)
        st.write(fig)
