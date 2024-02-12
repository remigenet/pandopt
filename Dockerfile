FROM python:3.12-bullseye as builder

RUN python -m pip install pip==24.0
RUN python -m pip install pipx==1.4.3
RUN pipx ensurepath
RUN pipx install poetry==1.7.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /aplo

COPY pyproject.toml poetry.lock ./
RUN touch README.md
RUN pipx run poetry install --no-root
RUN pipx run poetry add numpy
RUN pipx run poetry add pandas
RUN pipx run poetry add numba
#RUN pipx inject poetry poetry-plugin


#RUN --mount=type=cache,target=$POETRY_CACHE_DIR pipx run poetry install --no-root

#FROM python:3.12-bullseye as runtime

#ENV VIRTUAL_ENV=/aplo/.venv \
 #   PATH="/aplo/.venv/bin:$PATH:/aplo"

#COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY ./ /pandopt/

WORKDIR /pandopt
RUN pipx run poetry config pypi-token.pypi pypi-AgEIcHlwaS5vcmcCJDFiY2NjNzcwLWFmZDgtNDU4OS1iZjg0LTc1MjI1OTVlOGM0MwACKlszLCJkZDhkZjA4Ni0yN2UxLTRmNjEtYWZhNi1hMjc5N2MzNWU3YWIiXQAABiAf0GtvByTUtkF3pZhELDtQ3RZB_GceYI-Y0QubcFv7aA
RUN pipx run poetry build
RUN pipx run poetry publish

